import torch
import numpy as np
from .traffic_signal import TrafficSignal

def traffic_signal_feature(ts, num_green_phases, device):
    '''
    Generate feature for given traffic signal.

    Args:
        ts (TrafficSignal): traffic signal object to generate feature node for.
        num_green_phases (int): number of green phases, to guarentee feature for all nodes are same size.
        device (str): type of device to cast for returning tensor.
    '''
    phase_id = [1 if ts.green_phase == i else 0 for i in range(num_green_phases)]
    min_green = [0 if ts.time_since_last_phase_change < ts.min_green + ts.yellow_time else 1] 
    density = ts.get_lanes_density()
    queue = ts.get_lanes_queue()
    avg_density = [np.mean(density)]
    min_density = [np.min(density)]
    max_density = [np.max(density)]

    avg_queue = [np.mean(queue)]
    min_queue = [np.min(queue)]
    max_queue = [np.max(queue)]

    return torch.Tensor(phase_id + min_green \
                        + avg_density + min_density + max_density \
                        + avg_queue + min_queue + max_queue, device=device)


def batch_traffic_signal_feature(ts_list, ts_node_indx, num_nodes, device):
    '''
    Return feature tensor matrix for all TrafficSignals, returning of size NxD.

    Args:
        ts_list (list[TrafficSignal]): list of traffic signal object to generate feature node for.
        ts_node_indx (dict[str: int]): mapping of TrafficSignal id to associated node index.
        num_nodes (int): number of nodes in graph. Since graph includes dummy nodes with no mapping to TrafficSignal.
        device (str): type of device to cast for returning tensor.
    '''
    batch = [None for _ in range(num_nodes)]
    max_green_phases = np.max([ts.num_green_phases for ts in ts_list])
    feature_size = None
    for ts in ts_list:
        feature = traffic_signal_feature(ts, max_green_phases, device)
        batch[ts_node_indx[ts.id]] = feature
        if feature_size is None:
            feature_size = len(feature)

    assert feature_size != None

    result = torch.zeros([num_nodes, feature_size], dtype=torch.float64, device=device)
    for i in range(num_nodes):
        curr = batch[i]
        if curr is not None:
            result[i] = curr

    return result
