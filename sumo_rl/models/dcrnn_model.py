import torch
import torch.nn as nn
from dcrnn_cell import DCGRUCell
from base_model import BaseModel

class DCRNNEncoder(BaseModel):
    def __init__(self, input_dim, adj_mat, max_diffusion_step, hid_dim, num_nodes,
                 num_rnn_layers, filter_type):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self._num_rnn_layers = num_rnn_layers

        encoding_cells = list()

        # the first layer has different input_dim
        encoding_cells.append(DCGRUCell(input_dim=input_dim, num_units=hid_dim, adj_mat=adj_mat,
                                        max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes, filter_type=filter_type))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(DCGRUCell(input_dim=hid_dim, num_units=hid_dim, adj_mat=adj_mat,
                                            max_diffusion_step=max_diffusion_step,
                                            num_nodes=num_nodes, filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state):
        # inputs shape: (seq_len, batch, num_nodes, input_dim)
        # inputs to cell: (batch, num_nodes * input_dim)
        # init_hidden_state: (num_layers, batch, num_nodes*num_units)
        seq_length, batch_size = inputs.shape[:2]
        inputs = inputs.view(seq_length, batch_size, -1)  # (seq_len, batch, num_nodes * input_dim)

        current_inputs = inputs
        #final_hidden_states = []

        for i in range(self._num_rnn_layers):
            hidden_state = initial_hidden_state[i]
            layer_outputs = []

            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i](current_inputs[t], hidden_state)
                layer_outputs.append(hidden_state)

            #final_hidden_states.append(hidden_state)

            # Update current_inputs for the next layer
            current_inputs = torch.stack(layer_outputs,
                                         dim=0)  # Shape: (seq_length, batch_size, num_nodes * hid_dim)

        # output_hidden = torch.stack(final_hidden_states, dim=0)  # (num_layers, batch_size, num_nodes * hid_dim)
        # return output_hidden, current_inputs

        final_output = current_inputs[-1].view(batch_size, self._num_nodes,
                                               self._num_units)  # Shape: (batch_size, num_nodes, hid_dim)
        return final_output

