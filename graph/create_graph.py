from graph.graph_converter import construct_graph_representation
from graph.node_features import batch_traffic_signal_feature
from sumo_rl.environment.traffic_signal import TrafficSignal

def construct_graph_and_features(ts_list, device, observations=None):
  '''
  Returns:
    * node_features (Tensor): list of feature vectors for each node.
    * adj_list (Tensor): list of directed edges in graph, ex: [[0,1],[2,3]].
    * ts_indx (dict[str: int]): mapping of TrafficSignal id to node index.
    * lane_indx (dict[str: int]): mapping of lane id to adj_list index.
    * num_nodes (int): Total number of nodes in graph.

  Args:
    * ts_list (list[TrafficSignal]): list of TrafficSignals.
    * device (str): Type of device for Tensor.
  '''
  ts_indx, num_nodes, lane_indx, adj_list = construct_graph_representation(ts_list, device)
  node_features = batch_traffic_signal_feature(ts_list, ts_indx, num_nodes, device, observations)

  return node_features, adj_list, ts_indx, lane_indx, num_nodes



