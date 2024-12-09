import scipy.sparse as sp
from scipy.sparse import linalg
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import floyd_warshall
import torch_geometric
import networkx as nx
import numpy as np
import torch
import torch_geometric

def build_networkx_G(adj_list):
    G = nx.DiGraph()
    for u,v in adj_list:
        G.add_edge(u,v)
    return G

def get_laplacian_eigenvecs(G):
    G_undirected = G.to_undirected()
    laplacian = nx.laplacian_matrix(G_undirected).toarray()
    eigenvals, eigenvecs = np.linalg.eig(laplacian)
    return laplacian, eigenvals, eigenvecs

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

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

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)  # L is coo matrix
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    # L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='coo', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    # return L.astype(np.float32)
    return L.tocoo()

def construct_graph_representation(ts_list):
    '''
    Build graph representation of TrafficSignal traffic system.
    Returns:
        ts_idx: mapping of TrafficSignal id to associated node index.
        num_nodes: total number of nodes in graph.
        lanes_index: mapping of lane id to associated edge index in adj_list.
        adj_list (np.array): Adjancency list of size [|E|, 2].
        incoming_lane_ts (list[int]): containing the indices of traffic signals connecting to incoming virtual nodes.
        outgoing_lane_ts (list[int]): containing the indices of traffic signals connecting to outgoing virtual nodes.

    Args:
        ts_list (list[TrafficSignal]): list of TrafficSignal to build graph representation for.
    '''
    # collect traffic signal ids
    sort_ts_func = lambda ts: ts.id
    ts_idx = {ts.id: i for i, ts in enumerate(sorted(ts_list, key=sort_ts_func))}

    # collect all lane ids
    lanes = [ts.lanes + ts.out_lanes for ts in ts_list]
    lanes = [lane for ln in lanes for lane in ln]
    lanes_index = {ln_id: i for i, ln_id in enumerate(sorted(set(lanes)))}

    # calculate all ts_start, ts_end for all lanes
    adj_list = [[-1 for _ in range(2)] for _ in range(len(lanes_index))]

    # fill with additional dummy nodes
    for ts in ts_list:
        ts_id = ts_idx[ts.id]
        for in_edge in ts.lanes:
            in_edge_idx = lanes_index[in_edge]
            adj_list[in_edge_idx][1] = ts_id

        for out_edge in ts.out_lanes:
            out_edge_idx = lanes_index[out_edge]
            adj_list[out_edge_idx][0] = ts_id

    incoming_indx = len(ts_idx)
    outgoing_indx = incoming_indx + 1

    incoming_lane_ts = []
    outgoing_lane_ts = []

    # for unassigned positions, add dummy nodes
    for lane in adj_list:
        if lane[0] == -1 or lane[1] == -1:
            if lane[1] == -1:
                lane[1] = outgoing_indx
                outgoing_lane_ts.append(lane[0])

            else:
                lane[0] = incoming_indx
                incoming_lane_ts.append(lane[1])
    num_nodes = outgoing_indx + 1

    return ts_idx, num_nodes, lanes_index, np.array(adj_list), incoming_lane_ts, outgoing_lane_ts


def process_observation_buffer_with_graph(observation_buffer, ts_idx, max_lanes, num_nodes):
    """
    Process the observation buffer into a uniform tensor suitable for model input

    Args:
        observation_buffer (list[dict]): Observation buffer where each entry is a dict of
                                         {ts_id: observation}, and observation contains:
                                         {
                                             "density": np.array([...]), length=num_lanes
                                             "queue": np.array([...])
                                         }.
        ts_idx (dict): Mapping of TrafficSignal id to node index from `construct_graph_representation`.
        max_lanes (int): Maximum number of lanes across all traffic signals.
        num_nodes (int): including virtual nodes incoming/outgoing_index

    Returns:
        np.ndarray: shape [num_timesteps, num_nodes, feature_size]

    """
    num_timesteps = len(observation_buffer)
    #num_traffic_signals = len(ts_idx)
    feature_size = 2 * max_lanes  # density and queue for each lane
    processed_buffer = np.zeros((num_timesteps, num_nodes, feature_size), dtype=np.float32)

    for t_idx, timestep in enumerate(observation_buffer):
        for ts_id, obs in timestep.items():
            ts_index = ts_idx[ts_id]
            # Pad density and queue
            padded_density = np.pad(obs["density"], (0, max_lanes - len(obs["density"])), mode="constant")
            padded_queue = np.pad(obs["queue"], (0, max_lanes - len(obs["queue"])), mode="constant")
            # Combine features
            processed_buffer[t_idx, ts_index, :] = np.concatenate([padded_density, padded_queue])

    return processed_buffer


def calculate_lane_statistics(ts_list, sumo):
    """
    Calculate mean and standard deviation of lane lengths.

    Args:
        ts_list (list[TrafficSignal]): List of traffic signals for accessing lane lengths.
        sumo: SUMO connection object

    Returns:
        mean_length:
        std_length:
    """
    lanes = set(lane for ts in ts_list for lane in ts.lanes + ts.out_lanes)
    lengths = [sumo.lane.getLength(lane) for lane in lanes]
    return np.mean(lengths), np.std(lengths)


def construct_weighted_adjacency_matrix(ts_list, sumo, k=None, num_virtual_nodes=2, inlane_ts=None, outlane_ts=None):
    """
    Build a weighted adjacency matrix for the traffic signal system, including virtual nodes.
    Weights are computed as exp(-dist(vi, vj)^2 / sigma^2) based on shortest path distances over the road network.

    Args:
        ts_list (list[TrafficSignal]): List of TrafficSignal to build graph representation for.
        sumo: SUMO connection object
        k (float): Threshold distance for connecting nodes.
        num_virtual_nodes (int): Number of virtual nodes to include in the graph.
        inlane_ts (list[int]): containing the indices of traffic signals connecting to incoming virtual nodes.
        outlane_ts (list[int]): containing the indices of traffic signals connecting to outgoing virtual nodes.

    Returns:
        ts_idx: Mapping of TrafficSignal id to associated node index.
        weighted_adj_mx (torch.Tensor): Weighted adjacency matrix.
    """

    # Extract adjacency matrix based on road network distances
    num_ts = len(ts_list)
    road_adj_matrix = np.zeros((num_ts, num_ts))

    for i, ts1 in enumerate(ts_list):
        for j, ts2 in enumerate(ts_list):
            if i != j:
                # Distance between traffic signals is the shortest lane between them
                shared_lanes = set(ts1.out_lanes).intersection(set(ts2.lanes))
                if shared_lanes:
                    shortest_length = min(sumo.lane.getLength(lane_id) for lane_id in shared_lanes)
                    road_adj_matrix[i, j] = shortest_length

    # Compute shortest path distances using Floyd-Warshall algorithm
    shortest_path_distances = floyd_warshall(road_adj_matrix, directed=True)
    print("Shortest path distances:", shortest_path_distances)

    # Calculate lane length statistics
    _, sigma = calculate_lane_statistics(ts_list, sumo)

    # Apply weighting formula to compute the weighted adjacency matrix
    adj_mx = np.exp(-shortest_path_distances ** 2 / (sigma ** 2))
    if k is not None:
        adj_mx[shortest_path_distances > k] = 0  # Apply threshold

    # Add virtual nodes
    adj_mx = np.pad(adj_mx, ((0, num_virtual_nodes), (0, num_virtual_nodes)), mode="constant", constant_values=0)

    # Connect virtual nodes
    incoming_idx = len(ts_list)
    outgoing_idx = incoming_idx + 1
    for signal_idx in inlane_ts:
        adj_mx[incoming_idx, signal_idx] = 1.0

    for signal_idx in outlane_ts:
        adj_mx[signal_idx, outgoing_idx] = 1.0

    return adj_mx

def construct_binary_adjacency_matrix(ts_list, num_virtual_nodes=2, inlane_ts=None, outlane_ts=None):
    """
    Build a binary adjacency matrix for the traffic signal system, including virtual nodes.
    Edges exist between nodes with shared lanes or virtual node connections.

    Args:
        ts_list (list[TrafficSignal]): List of TrafficSignal to build graph representation for.
        num_virtual_nodes (int): Number of virtual nodes to include in the graph.
        inlane_ts (list[int]): Indices of traffic signals connecting to incoming virtual nodes.
        outlane_ts (list[int]): Indices of traffic signals connecting to outgoing virtual nodes.

    Returns:
        ts_idx: Mapping of TrafficSignal id to associated node index.
        binary_adj_mx (np.ndarray): Binary adjacency matrix.
    """
    # Initialize adjacency matrix
    num_ts = len(ts_list)
    binary_adj_mx = np.zeros((num_ts, num_ts), dtype=np.float64)

    # Populate adjacency matrix based on shared lanes
    for i, ts1 in enumerate(ts_list):
        for j, ts2 in enumerate(ts_list):
            if i != j:
                shared_lanes = set(ts1.out_lanes).intersection(set(ts2.lanes))
                if shared_lanes:
                    binary_adj_mx[i, j] = 1

    # Add virtual nodes
    binary_adj_mx = np.pad(binary_adj_mx, ((0, num_virtual_nodes), (0, num_virtual_nodes)), mode="constant", constant_values=0)

    # Connect virtual nodes
    incoming_idx = len(ts_list)
    outgoing_idx = incoming_idx + 1
    for signal_idx in inlane_ts:
        binary_adj_mx[incoming_idx, signal_idx] = 1

    for signal_idx in outlane_ts:
        binary_adj_mx[signal_idx, outgoing_idx] = 1

    binary_adj_mx += np.eye(binary_adj_mx.shape[0])  # Add self-loops to the adjacency matrix
    return binary_adj_mx


def create_action_mask(num_nodes, max_green_phases, valid_action_counts):
    """
    Create a mask for traffic node actions, where each node may have a different number of valid actions.
    The last two nodes are virtual nodes and have no valid actions.

    Args:
        num_nodes (int): Total number of nodes, including traffic nodes and virtual nodes.
        max_green_phases (int): Maximum number of actions across all nodes.
        valid_action_counts (list[int]): List of valid action counts for each traffic node.

    Returns:
        torch.Tensor: A binary mask of shape (num_nodes, max_green_phases).
    """
    mask = torch.zeros((num_nodes, max_green_phases), dtype=torch.float32)

    # Assign valid actions for each node
    for i, count in enumerate(valid_action_counts):
        mask[i, :count] = 1.0

    return mask

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