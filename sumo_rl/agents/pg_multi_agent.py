from ..models.util import process_observation_buffer_with_graph
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric

class PGMultiAgent:
    def __init__(self, ts_indx, edge_index, num_nodes, k, hops, models, device, model_name, gamma=0.99, lr=1e-4):
        self.models = models
        self.optimizers = {}
        self.ts_indx = ts_indx
        self.num_nodes = num_nodes
        self.edge_index = edge_index
        self.model_name = model_name
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.k = k
        self.hops = hops

        for ts_id, model in self.models.items():
            self.optimizers[ts_id] = optim.Adam(model.parameters(), lr=self.lr)

        if self.model_name == "dcrnn":
            any_model = next(iter(self.models.values()))
            if hasattr(any_model.encoder, '_hid_dim'):
                self.hid_dim = any_model.encoder._hid_dim
            else:
                self.hid_dim = 128
        else:
            self.hid_dim = 128

        self.last_k_observations = {ts: [] for ts in ts_indx.keys()}
        self.no_op = {ts: 0 for ts in ts_indx.keys()}
        self.max_lanes = None

    def update(self, new_obs):
        for ts in self.last_k_observations.keys():
            # Just storing obs, no gradient involved here:
            self.last_k_observations[ts].append(new_obs[ts])
            if len(self.last_k_observations[ts]) > self.k:
                self.last_k_observations[ts].pop(0)

    def process_observations_for_agent(self, agent_name):
        agent_obs_list = []
        for obs_dict in self.last_k_observations[agent_name]:
            agent_obs_list.append({agent_name: obs_dict})

        processed_obs = process_observation_buffer_with_graph(
            agent_obs_list,
            self.ts_indx,
            self.max_lanes,
            self.num_nodes
        )
        # This should return a numpy array, converting to tensor:
        return torch.tensor(processed_obs, dtype=torch.float32, device=self.device)

    def create_local_graph(self, ts_id, ts_idx, adj_list, k, max_lane):
        # Only used if model_name != "dcrnn"
        node_index = ts_idx[ts_id]

        st_nodes, st_edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=node_index, num_hops=k, edge_index=adj_list,
            relabel_nodes=False, flow="source_to_target"
        )

        ts_nodes, ts_edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=node_index, num_hops=k, edge_index=adj_list,
            relabel_nodes=False, flow="target_to_source"
        )

        subgraph_nodes = torch.cat((st_nodes, ts_nodes)).unique(sorted=False)
        subgraph_edge_index = torch.cat((st_edge_index, ts_edge_index), dim=1).unique(dim=1)

        nodes_mapping = {node.item(): i for i, node in enumerate(subgraph_nodes)}

        subgraph_edge_index = torch.stack([
            torch.tensor([nodes_mapping[u.item()] for u in subgraph_edge_index[0]]),
            torch.tensor([nodes_mapping[v.item()] for v in subgraph_edge_index[1]])
        ], dim=0)

        idx_to_ts_id = {v: k for k, v in ts_idx.items()}

        subgraph_features = torch.zeros((self.k, len(subgraph_nodes), 2*max_lane), dtype=torch.float32)

        for local_idx, node_idx in enumerate(subgraph_nodes):
            a_id = idx_to_ts_id.get(node_idx.item(), None)
            if a_id is None:
                continue
            node_obs = self.last_k_observations[a_id]

            for t, obs in enumerate(node_obs):
                # Observations are floats, no gradients:
                density = torch.tensor(obs["density"], dtype=torch.float32)
                queue = torch.tensor(obs["queue"], dtype=torch.float32)

                padded_density = torch.nn.functional.pad(density, (0, max_lane - len(density)))
                padded_queue = torch.nn.functional.pad(queue, (0, max_lane - len(queue)))

                subgraph_features[t, local_idx, :] = torch.cat((padded_density, padded_queue))

        return subgraph_features, subgraph_nodes, subgraph_edge_index

    def train(self, env, num_episodes):
        all_episode_rewards = []
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}")
            obs, _ = env.reset()
            agents = env.agents
            self.update(obs)

            sumo_env = env.aec_env.env.env.env
            max_lanes = max(len(ts.lanes) for ts in sumo_env.traffic_signals.values())
            self.max_lanes = max_lanes

            agent_experiences = {agent_name: {'log_probs': [], 'rewards': []} for agent_name in env.agents}

            sumo_env.fixed_ts = True
            done = False
            it = 0
            while not done:
                if it < self.k:
                    obs, _, _, _, _ = env.step(self.no_op)
                    self.update(obs)
                    it += 1
                    continue

                if it == self.k:
                    sumo_env.fixed_ts = False
                    for _, ts in sumo_env.traffic_signals.items():
                        ts.run_rl_agents()

                it += 1
                actions = {}
                for agent_name in env.agents:
                    model = self.models[agent_name]

                    if self.model_name == "dcrnn":
                        obs_tensor = self.process_observations_for_agent(agent_name).unsqueeze(1)
                        initial_hidden_state = torch.zeros((1, self.num_nodes * self.hid_dim), device=self.device)
                        logits = model(obs_tensor, initial_hidden_state)  # (1, num_nodes, max_green_phases)

                        agent_idx = self.ts_indx[agent_name]
                        agent_logits = logits[0, agent_idx].clone()  # avoid in-place issues

                        dist = torch.distributions.Categorical(logits=agent_logits)
                        action = dist.sample()
                        actions[agent_name] = action.item()
                        log_prob = dist.log_prob(action)
                        agent_experiences[agent_name]['log_probs'].append(log_prob)
                    else:
                        agents_features, subgraph_nodes, subgraph_edge_index = self.create_local_graph(
                            agent_name, self.ts_indx, self.edge_index, self.hops, max_lanes
                        )
                        agent_idx = self.ts_indx[agent_name]
                        logits = model(agents_features, subgraph_edge_index, agent_idx, subgraph_nodes)

                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        actions[agent_name] = action.item()
                        log_prob = dist.log_prob(action)
                        agent_experiences[agent_name]['log_probs'].append(log_prob)

                observations, rewards, terminations, truncations, infos = env.step(actions)
                self.update(observations)

                for agent_name in env.agents:
                    reward = rewards[agent_name]
                    agent_experiences[agent_name]['rewards'].append(reward)

                done = all(terminations.values()) or all(truncations.values())

            # End of episode: update policies and compute episode reward
            total_ep_reward = 0.0
            count_agents = 0
            for agent_name in agents:
                print("Agent {} finished after {} timesteps".format(agent_name, it))
                optimizer = self.optimizers[agent_name]
                optimizer.zero_grad()
                log_probs = agent_experiences[agent_name]['log_probs']
                rewards = agent_experiences[agent_name]['rewards']

                # Compute returns
                returns_list = []
                Gt = 0
                for r in reversed(rewards):
                    Gt = r + self.gamma * Gt
                    returns_list.insert(0, Gt)
                returns = torch.tensor(returns_list, dtype=torch.float32, device=self.device)

                # Normalize returns
                r_mean = returns.mean()
                r_std = returns.std() + 1e-8
                returns = (returns - r_mean) / r_std

                losses = []
                for log_prob, Gt_val in zip(log_probs, returns):
                    losses.append(-log_prob * Gt_val)

                policy_loss = torch.stack(losses).sum()
                policy_loss.backward()
                optimizer.step()

                ep_agent_reward = sum(rewards)
                total_ep_reward += ep_agent_reward
                count_agents += 1

                print(f"Episode {episode + 1}, Agent {agent_name}, Total Reward: {ep_agent_reward}, Loss: {policy_loss.item()}")

            # Aggregate episode reward from all agents (e.g., average)
            if count_agents > 0:
                total_ep_reward = total_ep_reward / count_agents
            else:
                total_ep_reward = 0.0

            all_episode_rewards.append(total_ep_reward)

        return all_episode_rewards

    def save_models(self, path):
        first_agent_id = list(self.ts_indx.keys())[0]
        torch.save(self.models[first_agent_id].state_dict(), path)
        print(f"Model saved at {path}")
