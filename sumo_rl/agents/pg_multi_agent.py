from ..models.transformer_model import PolicyNetwork
import torch_geometric

import torch
import torch.nn as nn
import torch.optim as optim

class PGMultiAgent:
    def __init__(self, ts_indx, edge_index, num_nodes, k, hops, model_args, device, gamma=0.99, lr=1e-4):
        self.models = {}
        self.optimizers = {}
        self.ts_indx = ts_indx
        self.num_nodes = num_nodes
        self.edge_index = edge_index  # [2, |E|]
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.k = k
        self.hops = hops

        for ts_id in ts_indx.keys():
            self.models[ts_id] = PolicyNetwork(model_args).to(device)
            self.optimizers[ts_id] = optim.Adam(self.models[ts_id].parameters(), lr=lr)

        self.last_k_observations = {ts: [] for ts in ts_indx.keys()}
        self.no_op = {ts: 0 for ts in ts_indx.keys()}

    def update(self, new_obs):
        for ts in self.last_k_observations.keys():
            self.last_k_observations[ts].append(new_obs[ts])
            if len(self.last_k_observations[ts]) > self.k:
                self.last_k_observations[ts].pop(0)

    def create_local_graph(self, ts_id, ts_idx, adj_list, k, max_lane):
        """
        Create a local graph for the given traffic signal, considering both directions.

        Args:
            ts_id (str): Traffic signal ID.
            ts_idx (dict): Mapping of traffic signal IDs to node indices.
            adj_list (torch.Tensor): Edge list of the graph. [2, |E|]
            k (int): Number of hops for the subgraph.
            max_lane (int): Maximum number of lanes to pad density/queue.

        Returns:
            subgraph_features (torch.Tensor): Features for the local subgraph nodes.
            subgraph_nodes (torch.Tensor): Indices of nodes in the subgraph.
            subgraph_edge_index (torch.Tensor): Edge index of the subgraph.
        """
        node_index = ts_idx[ts_id]

        st_nodes, st_edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=node_index, num_hops=k, edge_index=adj_list, relabel_nodes=False, flow="source_to_target"
        )

        ts_nodes, ts_edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=node_index, num_hops=k, edge_index=adj_list, relabel_nodes=False, flow="target_to_source"
        )

        subgraph_nodes = torch.cat((st_nodes, ts_nodes)).unique(sorted=False)
        subgraph_edge_index = torch.cat((st_edge_index, ts_edge_index), dim=1).unique(dim=1)

        nodes_mapping = {node.item(): i for i, node in enumerate(subgraph_nodes)}

        # Map edges using the node mapping
        subgraph_edge_index = torch.stack([
            torch.tensor([nodes_mapping[u.item()] for u in subgraph_edge_index[0]]),
            torch.tensor([nodes_mapping[v.item()] for v in subgraph_edge_index[1]])
        ], dim=0)

        # Aggregate features for the combined subgraph
        idx_to_ts_id = {v: k for k, v in ts_idx.items()}

        # create placeholder for subgraph_features: shape (num_timesteps, num_nodes, feature_size)
        subgraph_features = torch.zeros((k, len(subgraph_nodes), 2*max_lane), dtype=torch.float32)

        for local_idx, node_idx in enumerate(subgraph_nodes):
            ts_id = idx_to_ts_id.get(node_idx.item(), None)
            if ts_id is None:
                continue
            node_obs = self.last_k_observations[ts_id]

            for t, obs in enumerate(node_obs):
                # Pad density and queue features
                density = torch.tensor(obs["density"], dtype=torch.float32)
                queue = torch.tensor(obs["queue"], dtype=torch.float32)

                padded_density = torch.nn.functional.pad(density, (0, max_lane - len(density)))
                padded_queue = torch.nn.functional.pad(queue, (0, max_lane - len(queue)))

                # Concatenate density and queue features
                subgraph_features[t, local_idx, :] = torch.cat((padded_density, padded_queue))

        return subgraph_features, subgraph_nodes, subgraph_edge_index

    def train(self, env, num_episodes):
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}")
            obs, _ = env.reset()
            agents = env.agents
            self.update(obs)  # initial observation
            agent_experiences = {agent_name: {'log_probs': [], 'rewards': []} for agent_name in env.agents}
            sumo_env = env.aec_env.env.env.env
            sumo_env.fixed_ts = True  # warmup phase
            max_lanes = max(len(ts.lanes) for ts in sumo_env.traffic_signals.values())
            done = False
            it = 0
            while not done:
                # Warmup phase with default traffic light control
                if it < self.k:
                    obs, _, _, _, _ = env.step(self.no_op)  # Run default traffic light control
                    self.update(obs)
                    it += 1
                    continue

                # RL control starts after warmup
                if it == self.k:
                    sumo_env.fixed_ts = False  # Switch to RL control
                    for _, ts in sumo_env.traffic_signals.items():
                        ts.run_rl_agents()  # Activate RL agents

                it += 1
                actions = {}
                for agent_name in env.agents:
                    agent_idx = self.ts_indx[agent_name]
                    # Create local graph
                    agents_features, subgraph_nodes, subgraph_edge_index = self.create_local_graph(
                        agent_name, self.ts_indx, self.edge_index, self.hops, max_lanes
                    )

                    model = self.models[agent_name]
                    logits = model(agents_features, subgraph_edge_index, agent_idx, subgraph_nodes)

                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    actions[agent_name] = action.item()
                    log_prob = dist.log_prob(action)
                    agent_experiences[agent_name]['log_probs'].append(log_prob)

                # Step the environment
                observations, rewards, terminations, truncations, infos = env.step(actions)
                self.update(observations)  # Use observations instead of obs

                # Store rewards
                for agent_name in env.agents:
                    reward = rewards[agent_name]
                    agent_experiences[agent_name]['rewards'].append(reward)

                # Check if all agents are done
                done = all(terminations.values()) or all(truncations.values())

            # At the end of the episode, update the policy for each agent
            for agent_name in agents:
                print("Agent {} finished after {} timesteps".format(agent_name, it))
                model = self.models[agent_name]
                optimizer = self.optimizers[agent_name]
                optimizer.zero_grad()
                log_probs = agent_experiences[agent_name]['log_probs']
                rewards = agent_experiences[agent_name]['rewards']

                # Compute returns (discounted rewards)
                returns = []
                Gt = 0
                for r in reversed(rewards):
                    Gt = r + self.gamma * Gt
                    returns.insert(0, Gt)
                returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
                # Normalize returns
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                # Compute policy loss
                policy_loss = 0  # Scalar accumulation
                for log_prob, Gt in zip(log_probs, returns):
                    policy_loss += -log_prob * Gt
                # Backpropagate the policy loss
                policy_loss.backward()
                optimizer.step()
                print(f"Episode {episode + 1}, Agent {agent_name}, Total Reward: {sum(rewards)}, Loss: {policy_loss.item()}")

    def save_models(self, path):
        """
        Save the model of the first agent (or any agent) to the specified path.
        You can modify this method to save all agents if desired.
        """
        first_agent_id = list(self.ts_indx.keys())[0]
        torch.save(self.models[first_agent_id].state_dict(), path)
        print(f"Model saved at {path}")
