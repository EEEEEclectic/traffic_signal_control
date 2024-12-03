import torch
import torch_geometric

def get_neighbors(ts_index, adj_list, k):
  '''
  Use k_hop_subgraph (https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.k_hop_subgraph).
  Returns dictionary with tuple representing subgraph:
    * subset (LongTensor): the nodes involved in the subgraph
    * edge_index (LongTensor): the filtered edge_index connectivity
    * mapping (LongTensor): the mapping from node indices in node_idx to their new location
    * edge_mask (BoolTensor): edge mask indicating which edges were preserved
  
  Args:
    * ts_indx (dict[str: int]): mapping of TrafficSignal id to node index.
    * adj_list (Tensor): list of directed edges in graph.
    * k (int): value of k, for k-hop neighbors.
  '''

  return {ts_id: torch_geometric.utils.k_hop_subgraph(node_idx=int(node_index), 
                                                           num_hops=k, 
                                                           edge_index=adj_list, 
                                                           relabel_nodes=True) \
               for ts_id, node_index in ts_index.items()}

class LastKFeatures:
  '''
  Stores last k features for each TrafficSignal.
  '''

  def __init__(self, node_index_list, feature_shape, k):
    '''
    Args:
      node_index_list (list[int]): List of node index
      feature_shape (tuple(int)): shape of feature.
      k (int): k features to track
    '''
    self.k = k
    self.ts_features = {node_index: [torch.zeros(feature_shape) for _ in range(k)] for node_index in node_index_list }

  def update(self, features):
    '''
    Update self.ts_features with most recent features.

    Args:
      features (dict{int: tensor}): dictionary mapping feature for each node index.
    '''

    for ts_id, feature in features.items():
      self.ts_features[ts_id].insert(0, feature)
      self.ts_features[ts_id].pop()

    

if __name__ == "__main__":
  node_index_list = [0, 1, 2]
  k = 2
  last_k = LastKFeatures(node_index_list, (5,), k)
  print("STEP 1")
  print(f"ts_features: {last_k.ts_features}")
  
  features = {0: torch.ones(5), 1: torch.ones(5), 2: torch.ones(5)}
  last_k.update(features)
  print("STEP 2")
  print(f"ts_features: {last_k.ts_features}")
  
  features = {0: torch.full((5,), 2), 1: torch.full((5,), 2), 2: torch.full((5,), 2)}
  last_k.update(features)
  print("STEP 3")
  print(f"ts_features: {last_k.ts_features}")
  
  features = {0: torch.full((5,), 3), 1: torch.full((5,), 3), 2: torch.full((5,), 3)}
  last_k.update(features)
  print("STEP 4")
  print(f"ts_features: {last_k.ts_features}")

  print("Generate subgraphs")
  ts_index = {'ts0': 0, 'ts1': 1, 'ts2': 2, 'ts3': 3, 'ts4': 4, 'ts5': 5}
  adj_list = torch.LongTensor([[0,0,1,3,3,2,3,4,4,5],[1,3,0,0,2,3,4,3,5,4]])
  k=2
  res = get_neighbors(ts_index, adj_list, k)
  print(res)
  '''
  key: ts0
   value: (tensor([0, 1, 2, 3, 4]), tensor([[0, 0, 1, 3, 3, 2, 3, 4],
          [1, 3, 0, 0, 2, 3, 4, 3]]), tensor([0]), tensor([ True,  True,  True,  True,  True,  True,  True,  True, False, False]))
  key: ts1
   value: (tensor([0, 1, 3]), tensor([[0, 0, 1, 2],
          [1, 2, 0, 0]]), tensor([1]), tensor([ True,  True,  True,  True, False, False, False, False, False, False]))
  key: ts2
   value: (tensor([0, 2, 3, 4]), tensor([[0, 2, 2, 1, 2, 3],
          [2, 0, 1, 2, 3, 2]]), tensor([1]), tensor([False,  True, False,  True,  True,  True,  True,  True, False, False]))
  key: ts3
   value: (tensor([0, 1, 2, 3, 4, 5]), tensor([[0, 0, 1, 3, 3, 2, 3, 4, 4, 5],
          [1, 3, 0, 0, 2, 3, 4, 3, 5, 4]]), tensor([3]), tensor([True, True, True, True, True, True, True, True, True, True]))
  key: ts4
   value: (tensor([0, 2, 3, 4, 5]), tensor([[0, 2, 2, 1, 2, 3, 3, 4],
          [2, 0, 1, 2, 3, 2, 4, 3]]), tensor([3]), tensor([False,  True, False,  True,  True,  True,  True,  True,  True,  True]))
  key: ts5
   value: (tensor([3, 4, 5]), tensor([[0, 1, 1, 2],
          [1, 0, 2, 1]]), tensor([2]), tensor([False, False, False, False, False, False,  True,  True,  True,  True]))
  '''

