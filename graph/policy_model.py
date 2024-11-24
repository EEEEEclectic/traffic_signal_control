import torch
import numpy as np
from torch.nn import Linear, LeakyReLU, Sequential, ModuleList, LogSoftmax
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.aggr import MLPAggregation, MeanAggregation

class TransformerBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, heads=4):
    super(TransformerBlock, self).__init__()
    self.transformer = TransformerConv(in_channels=in_channels, out_channels=out_channels, heads=heads)
    self.relu = LeakyReLU()

  def forward(self, x, edge_index):
    x = self.transformer(x, edge_index)
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
        self.eigenvecs = args["eigenvecs"]
        self.eigenvec_len = self.eigenvecs.shape[0]
        self.num_layers = args.get("num_transformer_layers", 2)
        self.num_proj_layers = args.get("num_proj_layers", 2)

        assert self.num_layers > 1
        assert self.num_proj_layers > 1


        self.input_features = args["in_features"]
        self.hidden_features = args["hidden_features"]
        self.output_features = args["output_features"]
        # Add Transformer layers
        self.layers = ModuleList()
        self.layers.append(TransformerBlock(in_channels=self.input_features+self.eigenvec_len, out_channels=self.hidden_features, heads=4))
        for _ in range(self.num_layers-2):
          self.layers.append(TransformerBlock(in_channels=self.hidden_features+self.eigenvec_len, out_channels=self.hidden_features, heads=4))
        self.layers.append(TransformerConv(in_channels=self.hidden_features+self.eigenvec_len, out_channels=self.hidden_features, heads=4))

        # Add projection head for classification
        proj_head = []
        for _ in range(self.num_proj_layers-1):
          proj_head.append(Linear(in_features=self.hidden_features, out_features=self.hidden_features))
          proj_head.append(LeakyReLU())
        proj_head.append(Linear(in_features=self.hidden_features, out_features=self.output_features))

        self.proj_head = Sequential(*proj_head)
        self.softmax = LogSoftmax()

  def forward(self, node_features, edge_index, agent_index, num_green_phases, subgraph_indices):
    x = node_features
    pos = np.take(self.eigenvecs, subgraph_indices, dim=0)
    for layer in self.layers:
      x = torch.cat([x, pos], dim=1)
      x = layer(x)

    output = self.proj(x[agent_index])
    output = output[:num_green_phases] # perform masking to only legal actions
    probs = self.softmax(output)

    return probs
  
