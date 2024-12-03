import torch
from sumo_rl.environment.traffic_signal import TrafficSignal

def construct_graph_representation(ts_list, device):
    '''
    Build graph representation of TrafficSignal traffic system.
    Returns:
        ts_idx: mapping of TrafficSignal id to associated node index.
        num_nodes: total number of nodes in graph.
        lanes_index: mapping of lane id to associated edge index in adj_list.
        adj_list (torch.Tensor): Adjancency list of size [|E|, 2].

    Args:
        ts_list (list[TrafficSignal]): list of TrafficSignal to build graph representation for.
        device (str): Type of device for generating Tensor.
    '''
    # collect traffic signal ids
    sort_ts_func = lambda ts: ts.id
    ts_idx = {ts.id: i for i, ts in enumerate(sorted(ts_list, key=sort_ts_func))}

    # collect all lane ids
    lanes = [ts.lanes + ts.out_lanes for ts in ts_list]
    lanes = [lane for ln in lanes for lane in ln]
    lanes_index = {ln_id: i for i,ln_id in enumerate(sorted(set(lanes)))}

    # calculate all ts_start, ts_end for all lanes
    adj_list = [[-1 for _ in range(2)] for _ in range(len(lanes_index))]


    # Assign head, tail nodes to each edge.
    for ts in ts_list:
        ts_id = ts_idx[ts.id]
        for in_edge in ts.lanes:
            in_edge_idx = lanes_index[in_edge]
            adj_list[in_edge_idx][1] = ts_id
    
        for out_edge in ts.out_lanes:
            out_edge_idx = lanes_index[out_edge]
            adj_list[out_edge_idx][0] = ts_id

    clean_adj_list = []
    # for unassigned positions, remove edge from graph.
    for lane in adj_list:
        if lane[0] == -1 or lane[1] == -1:
            continue
        clean_adj_list.append(lane)
    num_nodes = len(clean_adj_list)

    return ts_idx, num_nodes, lanes_index, torch.LongTensor(clean_adj_list, device=device)
