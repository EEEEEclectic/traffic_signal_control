import torch
import numpy as np
from torch.nn import Linear, LeakyReLU, Sequential, ModuleList, LogSoftmax, BatchNorm1d
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.aggr import MLPAggregation, MeanAggregation


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.0):
        super(TransformerBlock, self).__init__()
        self.transformer = TransformerConv(
            in_channels=in_channels, out_channels=out_channels, heads=heads, dropout=dropout)
        self.norm = BatchNorm1d(num_features=out_channels)
        self.relu = LeakyReLU()
        self.proj = Linear(in_features=heads*out_channels,
                           out_features=out_channels)

    def forward(self, x, edge_index):
        x = self.transformer(x, edge_index)
        x = self.proj(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class PolicyNetwork(torch.nn.Module):
    def __init__(self, ts_map, traffic_system, args):
        '''
        Args:
        * ts_map (Dict{str: int}): mapping of traffic signal id to node index
        * traffic_system (Tensor[2,|E|]): adjacency list of entire traffic system
        * args (dict): arguments relating to traffic system. e.i: eigenvectors.
        * aggr (str): type of aggregation to use.
        '''
        super(PolicyNetwork, self).__init__()
        self.traffic_system = traffic_system
        self.ts_map = ts_map
        self.laplacian = args["laplacian_matrix"]
        self.eigenvecs = torch.FloatTensor(args["eigenvecs"])
        self.eigenvec_len = self.eigenvecs.shape[0]
        self.num_layers = args.get("num_transformer_layers", 2)
        self.num_proj_layers = args.get("num_proj_layers", 2)

        assert self.num_layers > 1
        assert self.num_proj_layers > 1

        self.input_features = args["in_features"]
        self.hidden_features = args["hidden_features"]
        self.output_features = args["output_features"]
        self.dropout = args.get("dropout", 0.0)

        # Add preprocessing layers
        self.preprocess = Sequential(
            Linear(in_features=self.input_features, out_features=self.hidden_features),
            LeakyReLU(),
            Linear(in_features=self.hidden_features, out_features=self.hidden_features))
        # Add Transformer layers
        self.layers = ModuleList()
        self.layers.append(TransformerBlock(in_channels=self.hidden_features+self.eigenvec_len,
                           out_channels=self.hidden_features, heads=4, dropout=self.dropout))
        prev = self.input_features
        for _ in range(self.num_layers-2):
            self.layers.append(TransformerBlock(in_channels=self.hidden_features+prev +
                               self.eigenvec_len, out_channels=self.hidden_features, heads=4, dropout=self.dropout))
            prev = self.hidden_features
        self.layers.append(TransformerConv(in_channels=self.hidden_features+prev +
                           self.eigenvec_len, out_channels=self.hidden_features, heads=1, dropout=self.dropout))

        # Add projection head for classification
        proj_head = []
        for _ in range(self.num_proj_layers-1):
            proj_head.append(
                Linear(in_features=self.hidden_features, out_features=self.hidden_features))
            proj_head.append(LeakyReLU())
        proj_head.append(Linear(in_features=self.hidden_features,
                         out_features=self.output_features))

        self.proj_head = Sequential(*proj_head)
        self.softmax = LogSoftmax()

    def forward(self, node_features, edge_index, agent_index, subgraph_indices):
        '''
        Args:
        * node_features: node features
        * edge_index: edge index describing graph
        * agent_index: agent to generate action for
        * subgraph_indices: list of node indices used in subgraph to add positional encoding
        '''
        x = node_features.float()
        x = self.preprocess(x)
        
        pos = np.take(self.eigenvecs, subgraph_indices.cpu().numpy(), axis=0)
        pos = torch.tensor(pos, dtype=torch.float32)

        if pos.shape[0] != x.shape[0]:
            pos = torch.nn.functional.pad(
                pos, (0, 0, 0, x.shape[0] - pos.shape[0]))

        prev = None
        for layer in self.layers:
            temp = x
            if prev is not None:
                x = torch.cat([x, prev], dim=1)
            prev = temp
            x = torch.cat([x, pos], dim=1)
            x = layer(x, edge_index)

        output = self.proj_head(x[[agent_index]])
        probs = self.softmax(output)

        return probs

    # def forward(self, node_features, edge_index, agent_index, subgraph_indices):
    #     '''
    #     Args:
    #     * node_features: node features
    #     * edge_index: edge index describing graph
    #     * agent_index: agent to generate action for
    #     * num_green_phases: number of green phases for agent, used to mask action space to legal actions
    #     * subtgraph_indices: list of node indices used in subgraph to add positional encoding
    #     '''
    #     x = node_features
    #     pos = np.take(self.eigenvecs, subgraph_indices, axis=0)
    #     prev = None
    #     for layer in self.layers:
    #         temp = x
    #         if prev != None:
    #             x = torch.cat([x, prev], dim=1)
    #         prev = temp
    #         x = torch.cat([x, pos], dim=1)
    #         x = layer(x, edge_index)

    #     output = self.proj_head(x[[agent_index]])
    #     probs = self.softmax(output)

    #     return probs
