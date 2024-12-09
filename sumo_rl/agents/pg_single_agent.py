import torch
import torch.optim as optim
from torch.distributions.categorical import Categorical
from ..models.util import process_observation_buffer_with_graph
from ..models.util import get_neighbors

class PGSingleAgent:
    def __init__(self, actor, k, ts_idx, device, num_nodes, max_lanes, model_name, lr=1e-3,
                 last_k_features=None, node_features=None, edge_index=None):
        self.hid_dim = actor.encoder._hid_dim if hasattr(actor, 'encoder') else 128
        self.actor = actor
        self.ts_idx = ts_idx
        self.device = device
        self.num_nodes = num_nodes
        self.max_lanes = max_lanes
        self.model_name = model_name
        self.optimizer = optim.Adam(list(actor.parameters()), lr=lr)
        self.log_probs = []
        self.rewards = []
        self.last_k_observations = []
        self.k = k
        self.last_k_features = last_k_features
        self.node_features = node_features
        self.edge_index = edge_index
        # Single agent setup: only one agent
        self.agent_name = list(self.ts_idx.keys())[0]
        self.agent_idx = self.ts_idx[self.agent_name]

    def process_observations(self):
        processed_obs = process_observation_buffer_with_graph(
            self.last_k_observations, self.ts_idx, self.max_lanes, self.num_nodes,
        )
        return torch.tensor(processed_obs, dtype=torch.float32).to(self.device)
    
    def update_last_k_features(self):
        """
        Update last_k_features using the most recent timestep's features from process_observations().
        Only call this if last_k_features is not None.
        """
        if self.last_k_features is None:
            return

        obs_tensor = self.process_observations()  # (timesteps, num_nodes, feature_dim)
        if obs_tensor.shape[0] > 0:
            current_features = obs_tensor[-1]  # (num_nodes, feature_dim)
            # Create a dict {node_idx: feature}
            features_dict = {}
            for node_id in range(self.num_nodes):
                features_dict[node_id] = current_features[node_id]
            self.last_k_features.update(features_dict)

    def get_agent_features_and_subgraph(self):

        # Now, we use the full tuple returned by k_hop_subgraph:
        subgraph_data = get_neighbors(self.ts_idx, self.edge_index, self.k)
        subgraph_nodes, subgraph_edge_index, mapping, edge_mask = subgraph_data[self.agent_name]
        if self.last_k_features is not None:
            node_list = list(self.ts_idx.values())
            all_nodes = range(self.num_nodes)
            feature_dim = self.last_k_features.ts_features[node_list[0]][0].shape[-1]
            agents_features = torch.zeros((self.k, self.num_nodes, feature_dim), device=self.device)
            for nid in all_nodes:
                if nid in self.last_k_features.ts_features:
                    node_feats = torch.stack(self.last_k_features.ts_features[nid], dim=0)  # (k, feature_dim)
                else:
                    node_feats = torch.zeros((self.k, feature_dim), device=self.device)
                agents_features[:, nid, :] = node_feats
        else:
            # If no last_k_features, fallback to using process_observations last step
            obs_tensor = self.process_observations()
            # obs_tensor: (num_timesteps, num_nodes, feature_dim)
            if obs_tensor.shape[0] > 0:
                # We have temporal dimension in obs_tensor
                agents_features = obs_tensor  # shape: (num_timesteps, num_nodes, feature_dim)
            else:
                # No observations, return a zero feature
                feature_dim = 2 * self.max_lanes
                # We must return (num_timesteps, num_nodes, feature_dim), at least 1 timestep
                agents_features = torch.zeros((1, self.num_nodes, feature_dim), device=self.device)

        edge_index_device = subgraph_edge_index.to(self.device)
        return agents_features, subgraph_nodes, edge_index_device


    # def get_agent_features_and_subgraph(self):
    #     # get_neighbors returns subgraph info
    #     subgraph_data = get_neighbors(self.ts_idx, self.edge_index, self.k)
    #     subgraph_indices = subgraph_data[self.agent_name][0]

    #     # If last_k_features is used (Colab logic):
    #     if self.last_k_features is not None:
    #         # Reconstruct agents_features from last_k_features
    #         feature_list = []
    #         for _, node_idx in self.ts_idx.items():
    #             # Use the most recent features: last_k_features.ts_features[node_idx][0]
    #             feature_list.append(torch.stack(self.last_k_features.ts_features[node_idx], dim=0))
    #         agents_features = torch.stack(feature_list, dim=1)
    #     else:
    #         # If no last_k_features, fallback to using process_observations last step
    #         obs_tensor = self.process_observations()
    #         # Take last timestep features if available
    #         if obs_tensor.shape[0] > 0:
    #             agents_features = obs_tensor[-1]  # shape: (num_nodes, feature_dim)
    #         else:
    #             # No observations, return a zero feature
    #             feature_dim = 2 * self.max_lanes
    #             agents_features = torch.zeros((self.num_nodes, feature_dim), device=self.device)

    #     if self.num_nodes > len(self.ts_idx):
    #         feature_dim = agents_features.size(-1)
    #         padding = torch.zeros((agents_features.size(0), self.num_nodes - len(self.ts_idx), feature_dim), device=self.device)
    #         agents_features = torch.cat([agents_features, padding], dim=1)

    #     agents_features = agents_features.to(self.device)
    #     subgraph_indices = subgraph_indices.to(self.device)
    #     edge_index_device = self.edge_index.to(self.device)
    #     return agents_features, subgraph_indices, edge_index_device

    def select_actions(self, logits):
        actions = {}
        for agent_name, agent_idx, logit in logits:
            action_distributions = torch.softmax(logit, dim=-1)
            log_prob_sum = 0
            dist = Categorical(action_distributions[agent_idx])
            action = dist.sample()
            actions[agent_name] = action.item()
            log_prob_sum += dist.log_prob(action)
        self.log_probs.append(log_prob_sum)
        return actions

    def compute_global_reward(self, rewards):
        # Single agent: just return the agent's reward
        return sum(rewards.values())

    def train(self, env, num_episodes):
        episode_rewards = []
        for episode in range(num_episodes):
            obs = env.reset()
            env.fixed_ts = True
            self.log_probs = []
            self.rewards = []
            self.last_k_observations = [obs]
            step_count = 0

            while True:
                if step_count < self.k:
                    obs, _, _, _ = env.step(action=None)
                    self.last_k_observations.append(obs)
                    step_count += 1
                    continue

                if step_count == self.k:
                    env.fixed_ts = False
                    for _, ts in env.traffic_signals.items():
                        ts.run_rl_agents()

                # Process observations
                # For DCRNN, we did obs_tensor and initial_hidden_state
                # For transformer, we must call get_agent_features_and_subgraph
                if self.model_name == "dcrnn":
                    obs_tensor = self.process_observations().unsqueeze(1)
                    initial_hidden_state = torch.zeros((1, self.num_nodes * self.hid_dim), device=self.device)
                    logits = self.actor(obs_tensor, initial_hidden_state).squeeze(0)
                else:
                    # transformer logic:
                    agents_features, subgraph_indices, edge_index_device = self.get_agent_features_and_subgraph()
                    # call the actor as in colab code
                    output = self.actor(agents_features, edge_index_device, self.agent_idx, subgraph_indices)
                    logits = output

                actions = self.select_actions(logits)
                obs, rewards, dones, info = env.step(actions)
                self.last_k_observations.append(obs)
                if len(self.last_k_observations) > self.k:
                    self.last_k_observations.pop(0)

                global_reward = self.compute_global_reward(rewards)
                self.rewards.append(global_reward)

                if dones["__all__"]:
                    break
                step_count += 1

            discounted_rewards = self._compute_discounted_rewards()
            policy_loss = self._compute_policy_loss(discounted_rewards)
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            total_ep_reward = sum(self.rewards)
            episode_rewards.append(total_ep_reward)
            print(f"Episode {episode}, Total Reward: {total_ep_reward}, Loss: {policy_loss.item()}")

        return episode_rewards

    def _compute_discounted_rewards(self, gamma=0.99):
        if len(self.rewards) == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)
        discounted_rewards = []
        cumulative_reward = 0.0
        for reward in reversed(self.rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=self.device)
        if discounted_rewards.numel() > 1:
            return (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        else:
            return discounted_rewards

    def _compute_policy_loss(self, discounted_rewards):
        policy_loss = torch.tensor(0.0, device=self.device)  # start as tensor
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss = policy_loss - (log_prob * reward)
        return policy_loss

    def save_models(self, path):
        torch.save(self.actor.state_dict(), path)
        print(f"Model saved at {path}")
