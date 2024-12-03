import numpy as np

def evaluate(env, model, epochs, return_all_rewards):
  '''
  Evaluate model by running  model over x epochs and return average reward across all epochs, and list of reward per epoch if requested.

  Args:
    model: Model to evaluate
    epochs (int): number of epochs to run model over
  '''
  reward = []

  observations = env.reset()
  while env.agents:
    actions = model(env)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    reward.append(np.mean([rewards[ts] for ts in rewards))

  if return_all_rewards:
    return np.mean(reward), reward
  return np.mean(reward)
