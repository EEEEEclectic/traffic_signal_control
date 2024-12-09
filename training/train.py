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
                                 get_laplacian_eigenvecs, build_networkx_G, create_action_mask, LastKFeatures)
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
    traffic_signals = [ts for _, ts in env.traffic_signals.items()]
    MAX_LANES = max(len(ts.lanes) for ts in traffic_signals)
    MAX_GREEN_PHASES = max(ts.num_green_phases for ts in traffic_signals)
    ts_phases = [ts.num_green_phases for ts in traffic_signals]

    ts_idx, num_nodes, lanes_index, adj_list, incoming_lane_ts, outgoing_lane_ts = construct_graph_representation(traffic_signals)
    adj_mx = construct_binary_adjacency_matrix(traffic_signals, num_virtual_nodes=2,
                                               inlane_ts=incoming_lane_ts, outlane_ts=outgoing_lane_ts)
    adj_mx = torch.tensor(adj_mx, dtype=torch.float32)

    encoder = DCRNNEncoder(
        input_dim=2 * MAX_LANES,
        adj_mat=adj_mx,
        max_diffusion_step=5,
        hid_dim=128,
        num_nodes=num_nodes,
        num_rnn_layers=1,
        filter_type="dual_random_walk",
        device=args.device
    )

    action_mask = create_action_mask(num_nodes, MAX_GREEN_PHASES, ts_phases)
    head = TLPhasePredictor(
        hid_dim=128,
        input_dim=2 * MAX_LANES,
        num_nodes=num_nodes,
        num_virtual_nodes=2,
        max_green_phases=MAX_GREEN_PHASES,
        mask=action_mask,
        device=args.device
    )

    model = TSModel(encoder, head)
    model = model.to(args.device)
    agent = PGSingleAgent(actor=model, ts_idx=ts_idx, device=args.device,
                          num_nodes=num_nodes, model_name=args.model_type, max_lanes=MAX_LANES, lr=args.lr, k=5)

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
    # Build graph representation
    traffic_signals = [ts for _, ts in env.aec_env.env.env.env.traffic_signals.items()]
    MAX_LANES = max(len(ts.lanes) for ts in traffic_signals)
    MAX_GREEN_PHASES = max(ts.num_green_phases for ts in traffic_signals)
    ts_phases = [ts.num_green_phases for ts in traffic_signals]

    ts_idx, num_nodes, lanes_index, adj_list, incoming_lane_ts, outgoing_lane_ts = construct_graph_representation(traffic_signals)
    adj_mx = construct_binary_adjacency_matrix(traffic_signals, num_virtual_nodes=2,
                                               inlane_ts=incoming_lane_ts, outlane_ts=outgoing_lane_ts)
    adj_mx = torch.tensor(adj_mx, dtype=torch.float32)  # CPU tensor for computations

    action_mask = create_action_mask(num_nodes, MAX_GREEN_PHASES, ts_phases).to(args.device)

    # Instead of creating encoder/head once, create them per agent:
    models = {}
    for ts_id in ts_idx.keys():
        # Create a fresh encoder and head for each agent
        agent_encoder = DCRNNEncoder(
            input_dim=2 * MAX_LANES,
            adj_mat=adj_mx,
            max_diffusion_step=5,
            hid_dim=128,
            num_nodes=num_nodes,
            num_rnn_layers=1,
            filter_type="dual_random_walk",
            device=args.device
        )

        agent_head = TLPhasePredictor(
            hid_dim=128,
            input_dim=2 * MAX_LANES,
            num_nodes=num_nodes,
            num_virtual_nodes=2,
            max_green_phases=MAX_GREEN_PHASES,
            mask=action_mask,
            device=args.device
        )

        agent_model = TSModel(agent_encoder, agent_head).to(args.device)
        models[ts_id] = agent_model

    PGMA = PGMultiAgent(
        ts_indx=ts_idx,
        edge_index=torch.tensor(adj_list.T, dtype=torch.long).to(args.device),
        num_nodes=num_nodes,
        k=2,
        hops=2,
        models=models,  # each agent now truly has its own independent model
        device=args.device,
        model_name=args.model_type,
        gamma=args.gamma,
        lr=args.lr
    )

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

def train_centralized_transformer(env, args, writer):
    traffic_signals = [ts for _, ts in env.traffic_signals.items()]
    MAX_LANES = max(len(ts.lanes) for ts in traffic_signals)
    MAX_GREEN_PHASES = max(ts.num_green_phases for ts in traffic_signals)
    ts_phases = [ts.num_green_phases for ts in traffic_signals]

    ts_idx, num_nodes, lanes_index, adj_list, incoming_lane_ts, outgoing_lane_ts = construct_graph_representation(traffic_signals)
    G = build_networkx_G(adj_list)
    laplacian_matrix, eigenvals, eigenvecs = get_laplacian_eigenvecs(G)
    action_mask = create_action_mask(num_nodes, MAX_GREEN_PHASES, ts_phases)
    edge_index = torch.tensor(adj_list.T, dtype=torch.long).to(args.device)

    # Single-agent environment reset: The env.reset() should return something like:
    # { "tl0": {"density": np.array(...), "queue": np.array(...)} }
    # We can use it directly.
    obs = env.reset()  # single-agent obs with the correct structure

    feature_dim = 2 * MAX_LANES
    k = 2
    last_k_features = LastKFeatures(node_index_list=range(num_nodes), feature_shape=(feature_dim,), k=k)

    model_args = {
        "laplacian_matrix": laplacian_matrix,
        "eigenvals": eigenvals,
        "eigenvecs": eigenvecs,
        "ts_indx": ts_idx,
        "in_features": feature_dim,
        "output_features": MAX_GREEN_PHASES,
        "num_transformer_layers": 2,
        "num_proj_layers": 2,
        "hidden_features": 128,
        "action_mask": action_mask,
        "device": args.device
    }

    policy = PolicyNetwork(model_args).to(args.device)
    agent = PGSingleAgent(
        actor=policy,
        ts_idx=ts_idx,
        device=args.device,
        model_name=args.model_type,  # even if model_name is "transformer", no if-block needed
        num_nodes=num_nodes,
        max_lanes=MAX_LANES,
        lr=args.lr,
        k=k,
        last_k_features=last_k_features,
        node_features=None,
        edge_index=edge_index
    )

    # Initialize agent's last_k_observations with the initial obs directly
    agent.last_k_observations = [obs]
    agent.update_last_k_features()

    weight_dir = os.path.join(args.out_dir, args.exp_name, "model_weight")
    os.makedirs(weight_dir, exist_ok=True)

    metrics = []
    for ep in range(args.episodes):
        obs = env.reset()  # single agent: returns {"tl0": {"density":..., "queue":...}}
        agent.log_probs = []
        agent.rewards = []
        agent.last_k_observations = [obs]
        agent.update_last_k_features()

        done = False
        step_count = 0

        it_border = 10
        while not done and step_count < it_border:
            if step_count < k:
                # Warm-up phase
                obs, _, _, _ = env.step(None)
                agent.last_k_observations.append(obs)
                agent.update_last_k_features()
                step_count += 1
                continue

            if step_count == k:
                env.fixed_ts = False
                for _, ts in env.traffic_signals.items():
                    ts.run_rl_agents()

            # Transformer logic:
            actions = {}
            logits = []
            for agent_name, agent_idx in ts_idx.items():
                agents_features, subgraph_indices, edge_index_device = agent.get_agent_features_and_subgraph(agent_name)
                logit = agent.actor(agents_features, edge_index_device, agent_idx, subgraph_indices)
                logits.append((agent_name, agent_idx, logit))

            # Select actions
            actions = agent.select_actions(logits)
            obs, reward, done, info = env.step(actions)
            done = done["A0"]  # Use A0 as proxy if done is true or still false
            agent.last_k_observations.append(obs)
            agent.update_last_k_features()
            agent.rewards.append(agent.compute_global_reward(reward))
            step_count += 1

        discounted_rewards = agent._compute_discounted_rewards()
        print(f"Discouinted rewards: {discounted_rewards}")
        if len(discounted_rewards) > 0:
            policy_loss = agent._compute_policy_loss(discounted_rewards)
            agent.optimizer.zero_grad()
            policy_loss.backward()
            agent.optimizer.step()
            total_ep_reward = sum(agent.rewards)
            mean_ep_rewards = np.mean(agent.rewards)
            mean_log_probs = np.mean(agent.log_probs.cpu())
        else:
            total_ep_reward = 0.0
            mean_ep_rewards = 0.0

        metrics.append(
            {
                "episode": {ep+1},
                "total_rewards": total_ep_reward,
                "mean_rewards": mean_ep_rewards,
                "mean_log_probs": mean_log_probs.cpu().numpy(),
                "loss": policy_loss.cpu().numpy()
            }
        )
        print(metrics)
        print(f"Episode {ep}, Reward: {total_ep_reward}")
        writer.add_scalar("Episode_Reward", total_ep_reward, ep)

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
    parallel = False if args.approach == "centralized" else True
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
