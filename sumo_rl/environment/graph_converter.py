import torch
from .traffic_signal import TrafficSignal

def construct_graph_representation(ts_list):
    '''
    Build graph representation of TrafficSignal traffic system.
    Returns:
        ts_idx: mapping of TrafficSignal id to associated node index.
        num_nodes: total number of nodes in graph.
        lanes_index: mapping of lane id to associated edge index in adj_list.
        adj_list (torch.Tensor): Adjancency list of size [|E|, 2].

    Args:
        ts_list (list[TrafficSignal]): list of TrafficSignal to build graph representation for.
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


    # fill with additional dummy nodes
    for ts in ts_list:
        ts_id = ts_idx[ts.id]
        for in_edge in ts.lanes:
            in_edge_idx = lanes_index[in_edge]
            adj_list[in_edge_idx][1] = ts_id
    
        for out_edge in ts.out_lanes:
            out_edge_idx = lanes_index[out_edge]
            adj_list[out_edge_idx][0] = ts_id

        print(ts.out_lanes)

    next_indx = len(ts_idx)
    # for unassigned positions, add dummy nodes
    for lane in adj_list:
        if lane[0] == -1 or lane[1] == -1:
            pos = 0
            if lane[1] == -1:
                pos = 1
            lane[pos] = next_indx
            next_indx += 1
    num_nodes = next_indx

    return ts_idx, num_nodes, lanes_index, torch.Tensor(adj_list)
    
