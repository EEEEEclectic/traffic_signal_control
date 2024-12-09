import numpy as np
from .util import *
import torch
import torch.nn as nn
from .base_model import BaseModel

class DiffusionGraphConv(BaseModel):
    def __init__(self, supports, input_dim, hid_dim, num_nodes, max_diffusion_step, output_dim, bias_start=0.0):
        super(DiffusionGraphConv, self).__init__()
        self.num_matrices = len(supports) * max_diffusion_step + 1
        input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step

        # Register each support as a buffer
        self._registered_supports = []
        for i, s in enumerate(supports):
            # s is a torch.sparse_coo_tensor
            self.register_buffer(f"support_{i}", s)
            self._registered_supports.append(getattr(self, f"support_{i}"))

        self.weight = nn.Parameter(torch.FloatTensor(input_size*self.num_matrices, output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(output_dim))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    def forward(self, inputs, state):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)

        x = inputs_and_state
        x = x.permute(1, 2, 0).reshape(self._num_nodes, -1)  # (num_nodes, input_size * batch_size)

        res = [x]
        if self._max_diffusion_step > 0:
            for support in self._registered_supports:
                x_k = x
                for _ in range(self._max_diffusion_step):
                    x_k = torch.sparse.mm(support, x_k)
                    res.append(x_k)

        x = torch.stack(res, dim=0)  # (num_matrices, nodes, input_size * bs)
        x = x.view(self.num_matrices, self._num_nodes, -1, batch_size).permute(3, 1, 2, 0)
        x = x.reshape(batch_size * self._num_nodes, -1)

        x = torch.matmul(x, self.weight)
        x = x + self.biases
        output = x.view(batch_size, -1)
        return output


class DCGRUCell(BaseModel):
    def __init__(self, input_dim, hid_dim, adj_mat, max_diffusion_step, num_nodes,
                 num_proj=None, activation=torch.tanh, use_gc_for_ru=True, filter_type='laplacian', device="cpu"):
        super(DCGRUCell, self).__init__()
        self._activation = activation
        self._num_nodes = num_nodes
        self._hid_dim = hid_dim
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._use_gc_for_ru = use_gc_for_ru

        # Build supports
        supports = []
        if isinstance(adj_mat, torch.Tensor):
            adj_mat = adj_mat.cpu().numpy()
        if filter_type == "laplacian":
            supports.append(calculate_scaled_laplacian(adj_mat, lambda_max=None))
        elif filter_type == "random_walk":
            supports.append(calculate_random_walk_matrix(adj_mat).T)
        elif filter_type == "dual_random_walk":
            supports.append(calculate_random_walk_matrix(adj_mat))
            supports.append(calculate_random_walk_matrix(adj_mat.T))
        else:
            supports.append(calculate_scaled_laplacian(adj_mat))

        # Convert supports to sparse tensors (CPU for now, model.to(device) later)
        device = torch.device(device=device)
        sparse_supports = [self._build_sparse_matrix(s).to(device) for s in supports]

        self.dconv_gate = DiffusionGraphConv(
            supports=sparse_supports,
            input_dim=input_dim, hid_dim=hid_dim,
            num_nodes=num_nodes, max_diffusion_step=max_diffusion_step,
            output_dim=hid_dim*2
        )

        self.dconv_candidate = DiffusionGraphConv(
            supports=sparse_supports,
            input_dim=input_dim, hid_dim=hid_dim,
            num_nodes=num_nodes, max_diffusion_step=max_diffusion_step,
            output_dim=hid_dim
        )

        if not use_gc_for_ru:
            self._fc = nn.Linear(input_dim + hid_dim, 2 * hid_dim)

        if num_proj is not None:
            self.project = nn.Linear(self._hid_dim, self._num_proj)

    def forward(self, inputs, state):
        inputs = inputs.to(self.weight.device) if hasattr(self, 'weight') else inputs
        state = state.to(self.weight.device) if hasattr(self, 'weight') else state

        fn = self.dconv_gate if self._use_gc_for_ru else self._fc

        gates = torch.sigmoid(fn(inputs, state))
        gates = gates.view(-1, self._num_nodes, 2 * self._hid_dim)

        r, u = torch.split(gates, self._hid_dim, dim=-1)
        r = r.reshape(-1, self._num_nodes * self._hid_dim)
        u = u.reshape(-1, self._num_nodes * self._hid_dim)

        c = self.dconv_candidate(inputs, r * state)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * state + (1 - u) * c
        output = new_state
        if self._num_proj is not None:
            batch_size = inputs.shape[0]
            output = self.project(new_state.view(-1, self._hid_dim)).view(batch_size, self._num_nodes * self._num_proj)

        return output, new_state

    @staticmethod
    def _build_sparse_matrix(L):
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse_coo_tensor(i, v, torch.Size(shape))
