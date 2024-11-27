import gymnasium as gym
import sumo_rl
import os
import torch
from custom_reward import custom_reward
from graph.create_graph import construct_graph_and_features
from graph.node_features import build_networkx_G, get_laplacian_eigenvecs, batch_traffic_signal_feature
from graph.k_neighbor import LastKFeatures, get_neighbors
from graph.policy_model import PolicyNetwork
import matplotlib.pyplot as plt
import networkx as nx


# from graph.create_graph import construct_graph_and_features
# from graph.node_features import build_networkx_G, get_laplacian_eigenvecs, batch_traffic_signal_feature

NET_FILE = './sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml'
ROUTE_FILE = './sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml'
OUTPUT_CSV = './output_csv/output.csv'
env = sumo_rl.parallel_env(
                net_file=NET_FILE,
                route_file=ROUTE_FILE,
                out_csv_name=OUTPUT_CSV,
                use_gui=False,
                num_seconds=3000,
                reward_fn=custom_reward
                )
observations = env.reset()

traffic_siginals = [ts for _, ts in env.aec_env.env.env.env.traffic_signals.items()]
node_features, adj_list, ts_indx, lane_indx, num_nodes = construct_graph_and_features(traffic_siginals, observations, "cpu")
adj_list = torch.unique(adj_list, dim=0).int() # Remove duplicate edges
G = build_networkx_G(adj_list.detach().numpy())
# nx.draw_networkx(G, arrows=True, with_labels=True)

laplacian_matrix, eigenvals, eigenvecs = get_laplacian_eigenvecs(G)
edge_index = adj_list.T.long()  # pytorch geometric takes in edge_index in shape (2,|E|). ex: [[0,1,0,2],[1,0,2,1]].

k = 2
last_k_features = LastKFeatures([i for i in range(num_nodes)], node_features[0].shape, k)
last_k_features.update({ node_index: node_features[node_index] for _, node_index in ts_indx.items()})

model_args = {
    "laplacian_matrix": laplacian_matrix,
    "eigenvals": eigenvals,
    "eigenvecs": eigenvecs,
    "ts_indx": ts_indx,
    "last_k_features": last_k_features,
    "in_features": node_features.shape[1],
    "output_features": 8,
    # Network hyperparameter
    "num_transformer_layers": 2,
    "num_proj_layers": 2,
    "hidden_features": 128
}

model = PolicyNetwork(ts_indx, edge_index, model_args)


def get_agent_i_features_and_subgraph(agent_index):
    subgraph_data = get_neighbors(ts_indx, edge_index, k)
    subgraph_indices = subgraph_data[list(ts_indx.keys())[agent_index]][0]
    agents_features = torch.concat([last_k_features.ts_features[node_idx][0].unsqueeze(0) for _, node_idx in ts_indx.items()])
    agents_features = torch.concat([agents_features, torch.zeros((num_nodes - len(ts_indx), node_features.size(1)))])
    
    return agents_features, subgraph_indices

num_iteration = 100
for i in range(len(num_iteration)):
    actions = {}
    for agent_name in env.agents:
        agent_idx = ts_indx[agent_name]
        agents_features, subgraph_indices = get_agent_i_features_and_subgraph(agent_idx)
        output = model(agents_features, edge_index, agent_idx, subgraph_indices)
        actions[agent_name] = int(torch.argmax(output))
    
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(rewards)
    print(observations)
    
    # Learn policy here
