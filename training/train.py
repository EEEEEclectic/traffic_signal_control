# train_rl.py
import os
import sys
import torch
import numpy as np
import csv
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cli_train import parse_args, save_config
from sumo_rl.environment.env import SumoEnvironment
from sumo_rl import parallel_env
from sumo_rl.models.util import (construct_graph_representation, construct_binary_adjacency_matrix,
                                 get_laplacian_eigenvecs, build_networkx_G, create_action_mask)
from sumo_rl.models.dcrnn_model import DCRNNEncoder, TLPhasePredictor, TSModel
from sumo_rl.models.transformer_model import PolicyNetwork
from sumo_rl.agents.pg_single_agent import PGSingleAgent
from sumo_rl.agents.pg_multi_agent import PGMultiAgent

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_env(args, parallel=True):
    if parallel:
        env = parallel_env(
            net_file=args.net_file,
            route_file=args.route_file,
            out_csv_name=None,
            use_gui=False,
            num_seconds=args.num_seconds,
            begin_time=args.begin_time,
            fixed_ts=args.fixed_ts,
            reward_fn=args.reward_fn
        )
    else:
        env = SumoEnvironment(
            net_file=args.net_file,
            route_file=args.route_file,
            out_csv_name=None,
            use_gui=False,
            num_seconds=args.num_seconds,
            begin_time=args.begin_time,
            fixed_ts=args.fixed_ts,
            reward_fn=args.reward_fn
        )
    return env


def train_centralized_dcrnn(env, args, writer):
    traffic_signals = [ts for _, ts in env.aec_env.env.env.env.traffic_signals.items()]
    MAX_LANES = max(len(ts.lanes) for ts in traffic_signals)
    MAX_GREEN_PHASES = max(ts.num_green_phases for ts in traffic_signals)
    ts_phases = [ts.num_green_phases for ts in traffic_signals]

    ts_idx, num_nodes, lanes_index, adj_list, incoming_lane_ts, outgoing_lane_ts = construct_graph_representation(traffic_signals)
    adj_mx = construct_binary_adjacency_matrix(traffic_signals, num_virtual_nodes=2,
                                               inlane_ts=incoming_lane_ts, outlane_ts=outgoing_lane_ts)

    encoder = DCRNNEncoder(
        input_dim=2 * MAX_LANES,
        adj_mat=adj_mx,
        max_diffusion_step=5,
        hid_dim=128,
        num_nodes=num_nodes,
        num_rnn_layers=1,
        filter_type="dual_random_walk"
    ).to(args.device)

    action_mask = create_action_mask(num_nodes, MAX_GREEN_PHASES, ts_phases)
    head = TLPhasePredictor(
        hid_dim=128,
        input_dim=2 * MAX_LANES,
        num_nodes=num_nodes,
        num_virtual_nodes=2,
        max_green_phases=MAX_GREEN_PHASES,
        mask=action_mask
    ).to(args.device)

    model = TSModel(encoder, head).to(args.device)
    agent = PGSingleAgent(actor=model, ts_idx=ts_idx, device=args.device,
                          num_nodes=num_nodes, max_lanes=MAX_LANES, lr=args.lr, k=5)

    # CHANGED: Create model_weight directory
    weight_dir = os.path.join(args.out_dir, args.exp_name, "model_weight")
    os.makedirs(weight_dir, exist_ok=True)

    rewards_all = []
    for ep in range(args.episodes):
        episode_rewards = agent.train(env, num_episodes=1)
        ep_reward = episode_rewards[0]
        rewards_all.append(ep_reward)
        print(f"Episode {ep}, Reward: {ep_reward}")
        writer.add_scalar("Episode_Reward", ep_reward, ep)

        # CHANGED: Save model every 10 episodes
        if (ep+1) % 10 == 0:
            filename = f"{args.model_type}_{args.approach}_lr{args.lr}_ep{ep+1}.pth"
            save_path = os.path.join(weight_dir, filename)
            agent.save_models(save_path)

    # Save final model
    final_filename = f"{args.model_type}_{args.approach}_lr{args.lr}_ep{args.episodes}_final.pth"
    final_save_path = os.path.join(weight_dir, final_filename)
    agent.save_models(final_save_path)

    return rewards_all


def train_decentralized_dcrnn(env, args, writer):
    print("Warning: Decentralized DCRNN logic not fully implemented, using centralized as placeholder.")
    return train_centralized_dcrnn(env, args, writer)


def train_centralized_transformer(env, args, writer):
    traffic_signals = [ts for _, ts in env.aec_env.env.env.env.traffic_signals.items()]
    MAX_LANES = max(len(ts.lanes) for ts in traffic_signals)
    MAX_GREEN_PHASES = max(ts.num_green_phases for ts in traffic_signals)
    ts_phases = [ts.num_green_phases for ts in traffic_signals]

    ts_idx, num_nodes, lanes_index, adj_list, incoming_lane_ts, outgoing_lane_ts = construct_graph_representation(traffic_signals)
    G = build_networkx_G(adj_list.T)
    laplacian_matrix, eigenvals, eigenvecs = get_laplacian_eigenvecs(G)
    action_mask = create_action_mask(num_nodes, MAX_GREEN_PHASES, ts_phases)

    model_args = {
        "laplacian_matrix": laplacian_matrix,
        "eigenvals": eigenvals,
        "eigenvecs": eigenvecs,
        "ts_indx": ts_idx,
        "in_features": 2*MAX_LANES,
        "output_features": MAX_GREEN_PHASES,
        "num_transformer_layers": 2,
        "num_proj_layers": 2,
        "hidden_features": 128,
        "action_mask": action_mask
    }

    policy = PolicyNetwork(model_args).to(args.device)
    agent = PGSingleAgent(actor=policy, ts_idx=ts_idx, device=args.device,
                          num_nodes=num_nodes, max_lanes=MAX_LANES, lr=args.lr, k=2)

    weight_dir = os.path.join(args.out_dir, args.exp_name, "model_weight")
    os.makedirs(weight_dir, exist_ok=True)

    rewards_all = []
    for ep in range(args.episodes):
        episode_rewards = agent.train(env, num_episodes=1)
        ep_reward = episode_rewards[0]
        rewards_all.append(ep_reward)
        print(f"Episode {ep}, Reward: {ep_reward}")
        writer.add_scalar("Episode_Reward", ep_reward, ep)

        if (ep+1) % 10 == 0:
            filename = f"{args.model_type}_{args.approach}_lr{args.lr}_ep{ep+1}.pth"
            save_path = os.path.join(weight_dir, filename)
            agent.save_models(save_path)

    final_filename = f"{args.model_type}_{args.approach}_lr{args.lr}_ep{args.episodes}_final.pth"
    final_save_path = os.path.join(weight_dir, final_filename)
    agent.save_models(final_save_path)

    return rewards_all


def train_decentralized_transformer(env, args, writer):
    traffic_signals = [ts for _, ts in env.aec_env.env.env.env.traffic_signals.items()]
    MAX_LANES = max(len(ts.lanes) for ts in traffic_signals)
    MAX_GREEN_PHASES = max(ts.num_green_phases for ts in traffic_signals)
    ts_phases = [ts.num_green_phases for ts in traffic_signals]

    ts_idx, num_nodes, lanes_index, adj_list, incoming_lane_ts, outgoing_lane_ts = construct_graph_representation(traffic_signals)
    G = build_networkx_G(adj_list.T)
    laplacian_matrix, eigenvals, eigenvecs = get_laplacian_eigenvecs(G)
    action_mask = create_action_mask(num_nodes, MAX_GREEN_PHASES, ts_phases)

    model_args = {
        "laplacian_matrix": laplacian_matrix,
        "eigenvals": eigenvals,
        "eigenvecs": eigenvecs,
        "ts_indx": ts_idx,
        "in_features": 2*MAX_LANES,
        "output_features": MAX_GREEN_PHASES,
        "num_transformer_layers": 2,
        "num_proj_layers": 2,
        "hidden_features": 128,
        "action_mask": action_mask
    }

    PGMA = PGMultiAgent(ts_indx=ts_idx, edge_index=torch.tensor(adj_list.T, dtype=torch.long),
                        num_nodes=num_nodes, k=2, hops=2, model_args=model_args,
                        device=args.device, gamma=args.gamma, lr=args.lr)

    weight_dir = os.path.join(args.out_dir, args.exp_name, "model_weight")
    os.makedirs(weight_dir, exist_ok=True)

    rewards_all = []
    for ep in range(args.episodes):
        episode_rewards = PGMA.train(env, num_episodes=1)
        ep_reward = episode_rewards[0]
        rewards_all.append(ep_reward)
        print(f"Episode {ep}, Reward: {ep_reward}")
        writer.add_scalar("Episode_Reward", ep_reward, ep)

        if (ep+1) % 10 == 0:
            filename = f"{args.model_type}_{args.approach}_lr{args.lr}_ep{ep+1}.pth"
            save_path = os.path.join(weight_dir, filename)
            PGMA.save_models(save_path)

    final_filename = f"{args.model_type}_{args.approach}_lr{args.lr}_ep{args.episodes}_final.pth"
    final_save_path = os.path.join(weight_dir, final_filename)
    PGMA.save_models(final_save_path)

    return rewards_all


if __name__ == "__main__":
    args = parse_args()
    save_config(args)
    parallel = True if args.approach == "decentralized" else False
    env = create_env(args, parallel=parallel)

    log_dir = os.path.join(args.out_dir, args.exp_name, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    if args.model_type == "dcrnn":
        if args.approach == "centralized":
            rewards_all = train_centralized_dcrnn(env, args, writer)
        else:
            rewards_all = train_decentralized_dcrnn(env, args, writer)
    else:
        if args.approach == "centralized":
            rewards_all = train_centralized_transformer(env, args, writer)
        else:
            rewards_all = train_decentralized_transformer(env, args, writer)

    writer.close()
    env.close()

    # Save rewards to CSV
    csv_path = os.path.join(args.out_dir, args.exp_name, "rewards.csv")
    with open(csv_path, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["episode", "reward"])
        for i, r in enumerate(rewards_all):
            writer_csv.writerow([i, r])
