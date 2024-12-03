from pettingzoo.test import api_test, parallel_api_test
from .create_graph import construct_graph_and_features
import torch

import sumo_rl


def train_model(model):
  '''
  Skeleton code to build training for model.
  '''
  env = sumo_rl.env(
        net_file="sumo_rl/nets/4x4-Lucas/4x4.net.xml",
        route_file="sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        out_csv_name="outputs/4x4grid/test",
        use_gui=False,
        num_seconds=100,
    )
  ts_list = list(env.traffic_signals.keys())
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  node_features, adj_list, ts_indx, lane_indx, num_nodes = construct_graph_and_features(ts_list, device)

  observations = env.reset()
  while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)

  
