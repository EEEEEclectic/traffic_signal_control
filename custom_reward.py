import numpy as np
GAMMA = 0.2
def custom_reward(traffic_signal):
    # return traffic_signal.get_total_queued() + (GAMMA * traffic_signal.get)
    wait_per_lanes = np.sum(np.sum(traffic_signal.get_max_cumulative_waiting_time_of_the_first_vehicles_per_lane()))
    queue_per_lanes = np.sum(traffic_signal.get_lanes_queue())
    return float(queue_per_lanes + (GAMMA * wait_per_lanes))