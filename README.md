# Traffic Signal Control via Multi Agent Graph-based Reinforcement Learning

This repository is based on the `sumo-rl` framework and extended with additional functionalities to handle graph-based representations of traffic networks and apply policy learning methods such as DCRNN or transformer-based models.

## Contents

- [Project Description](#project-description)
- [Repository Structure](#repository-structure)
  - [Graph Scripts](#graph-scripts)
  - [Policy Learning Scripts](#policy-learning-scripts)
  - [Experiment and Training Scripts](#experiment-and-training-scripts)
- [How to Run Experiments](#how-to-run-experiments)
  - [Prerequisites](#prerequisites)
  - [Example: Training Centralized DCRNN Policy](#example-training-centralized-dcrnn-policy)
- [Original SUMO-RL Documentation](#original-sumo-rl-documentation)
- [Citing](#citing)

## Project Description

This project aims to explore and evaluate graph-based reinforcement learning methods for traffic signal control. We integrate approaches like DCRNN and transformer-based models within the SUMO-RL environment and focus on capturing spatio-temporal dependencies in the traffic network via graph structures.

**Key goals:**
- Integrate graph-based representations to capture spatial and temporal correlation in the road network.

## Repository Structure

Below is an organized overview of important scripts and directories:
```
.
├─ sumo_rl/                      # Base SUMO-RL environment code
│   ├─ environment/              # Environment setup, traffic signal definitions
│   ├─ models/                   # Graph model class definition with utilities
│   ├─ agents/                   # Policy learning agents
├─ experiments/
│   ├─ centralised_dcrnn.py      # Centralized DCRNN training
│   ├─ decentralised_control.py  # Decentralized Transformer-based training
│   └─ ...
├─ outputs/
└─ README.md                     # This README
```


### Graph Scripts

- **`sumo_rl/models/base_model.py`**: Base class for all models.
- **`sumo_rl/models/dcrnn_cell.py`**: The model blocks for DCRNN model.
- **`sumo_rl/models/dcrnn_model.py`**: The DCRNN encoder and decoder model.
- **`sumo_rl/models/transformer_model.py`**: The graph transformer model.
- **`sumo_rl/models/util.py`**: Collection of graph operation fucntions including graph construction, retrieving k-hop neighbors, and temporal graph preprocessing. 

### Policy Learning Scripts

- **`sumo_rl/agents/pg_sigle_agent.py`**: Single agent policy learning script.
- **`sumo_rl/agents/pg_multi_agent.py`**: Multi agent policy learning script.

### Experiment and Training Scripts

- **`experiments/`**: Training scripts to run different models. 

## How to Run Experiments

### Prerequisites

1. **Prerequisite**: Install prerequisite according to the [Original SUMO-RL Documentation](#original-sumo-rl-documentation) instruction.
2. **Libraries**:
  - pytorch-geometric

### Example: Training Centralized DCRNN Policy

```bash
python experiments/centralised_dcrnn.py
```

## Original SUMO-RL Documentation

<img src="docs/_static/logo.png" align="right" width="30%"/>

[![DOI](https://zenodo.org/badge/161216111.svg)](https://zenodo.org/doi/10.5281/zenodo.10869789)
[![tests](https://github.com/LucasAlegre/sumo-rl/actions/workflows/linux-test.yml/badge.svg)](https://github.com/LucasAlegre/sumo-rl/actions/workflows/linux-test.yml)
[![PyPI version](https://badge.fury.io/py/sumo-rl.svg)](https://badge.fury.io/py/sumo-rl)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](https://github.com/LucasAlegre/sumo-rl/blob/main/LICENSE)

# Traffic Signal Control Using Decentralised MARL via Graph
<!-- start intro -->

TODO: Add more to README.

SUMO-RL provides a simple interface to instantiate Reinforcement Learning (RL) environments with [SUMO](https://github.com/eclipse/sumo) for Traffic Signal Control.

Goals of this repository:
- Provide a simple interface to work with Reinforcement Learning for Traffic Signal Control using SUMO
- Support Multiagent RL
- Compatibility with gymnasium.Env and popular RL libraries such as [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and [RLlib](https://docs.ray.io/en/main/rllib.html)
- Easy customisation: state and reward definitions are easily modifiable

The main class is [SumoEnvironment](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py).
If instantiated with parameter 'single-agent=True', it behaves like a regular [Gymnasium Env](https://github.com/Farama-Foundation/Gymnasium).
For multiagent environments, use [env](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py) or [parallel_env](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py) to instantiate a [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) environment with AEC or Parallel API, respectively.
[TrafficSignal](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/traffic_signal.py) is responsible for retrieving information and actuating on traffic lights using [TraCI](https://sumo.dlr.de/wiki/TraCI) API.

For more details, check the [documentation online](https://lucasalegre.github.io/sumo-rl/).

<!-- end intro -->

## Install

<!-- start install -->

### Install SUMO latest version:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```
Don't forget to set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```
Important: for a huge performance boost (~8x) with Libsumo, you can declare the variable:
```bash
export LIBSUMO_AS_TRACI=1
```
Notice that you will not be able to run with sumo-gui or with multiple simulations in parallel if this is active ([more details](https://sumo.dlr.de/docs/Libsumo.html)).

### Install SUMO-RL

Stable release version is available through pip
```bash
pip install sumo-rl
```

Alternatively, you can install using the latest (unreleased) version
```bash
git clone https://github.com/LucasAlegre/sumo-rl
cd sumo-rl
pip install -e .
```

<!-- end install -->

## MDP - Observations, Actions and Rewards

### Observation

<!-- start observation -->

The default observation for each traffic signal agent is a vector:
```python
    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]
```
- ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
- ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
- ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
- ```lane_i_queue```is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

You can define your own observation by implementing a class that inherits from [ObservationFunction](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/observations.py) and passing it to the environment constructor.

<!-- end observation -->

### Action

<!-- start action -->

The action space is discrete.
Every 'delta_time' seconds, each traffic signal agent can choose the next green phase configuration.

E.g.: In the [2-way single intersection](https://github.com/LucasAlegre/sumo-rl/blob/main/experiments/dqn_2way-single-intersection.py) there are |A| = 4 discrete actions, corresponding to the following green phase configurations:

<p align="center">
<img src="docs/_static/actions.png" align="center" width="75%"/>
</p>

Important: every time a phase change occurs, the next phase is preeceded by a yellow phase lasting ```yellow_time``` seconds.

<!-- end action -->

### Rewards

<!-- start reward -->

The default reward function is the change in cumulative vehicle delay:

<p align="center">
<img src="docs/_static/reward.png" align="center" width="25%"/>
</p>

That is, the reward is how much the total delay (sum of the waiting times of all approaching vehicles) changed in relation to the previous time-step.

You can choose a different reward function (see the ones implemented in [TrafficSignal](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/traffic_signal.py)) with the parameter `reward_fn` in the [SumoEnvironment](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py) constructor.

It is also possible to implement your own reward function:

```python
def my_reward_fn(traffic_signal):
    return traffic_signal.get_average_speed()

env = SumoEnvironment(..., reward_fn=my_reward_fn)
```

<!-- end reward -->

## API's (Gymnasium and PettingZoo)

### Gymnasium Single-Agent API

<!-- start gymnasium -->

If your network only has ONE traffic light, then you can instantiate a standard Gymnasium env (see [Gymnasium API](https://gymnasium.farama.org/api/env/)):
```python
import gymnasium as gym
import sumo_rl
env = gym.make('sumo-rl-v0',
                net_file='path_to_your_network.net.xml',
                route_file='path_to_your_routefile.rou.xml',
                out_csv_name='path_to_output.csv',
                use_gui=True,
                num_seconds=100000)
obs, info = env.reset()
done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
```

<!-- end gymnasium -->

### PettingZoo Multi-Agent API

<!-- start pettingzoo -->

For multi-agent environments, you can use the PettingZoo API (see [Petting Zoo API](https://pettingzoo.farama.org/api/parallel/)):

```python
import sumo_rl
env = sumo_rl.parallel_env(net_file='nets/RESCO/grid4x4/grid4x4.net.xml',
                  route_file='nets/RESCO/grid4x4/grid4x4_1.rou.xml',
                  use_gui=True,
                  num_seconds=3600)
observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}  # this is where you would insert your policy
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

<!-- end pettingzoo -->

### RESCO Benchmarks

In the folder [nets/RESCO](https://github.com/LucasAlegre/sumo-rl/tree/main/sumo_rl/nets/RESCO) you can find the network and route files from [RESCO](https://github.com/jault/RESCO) (Reinforcement Learning Benchmarks for Traffic Signal Control), which was built on top of SUMO-RL. See their [paper](https://people.engr.tamu.edu/guni/Papers/NeurIPS-signals.pdf) for results.

<p align="center">
<img src="sumo_rl/nets/RESCO/maps.png" align="center" width="60%"/>
</p>

### Experiments

Check [experiments](https://github.com/LucasAlegre/sumo-rl/tree/main/experiments) for examples on how to instantiate an environment and train your RL agent.

### [Q-learning](https://github.com/LucasAlegre/sumo-rl/blob/main/agents/ql_agent.py) in a one-way single intersection:
```bash
python experiments/ql_single-intersection.py
```

### [RLlib PPO](https://docs.ray.io/en/latest/_modules/ray/rllib/algorithms/ppo/ppo.html) multiagent in a 4x4 grid:
```bash
python experiments/ppo_4x4grid.py
```

### [stable-baselines3 DQN](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py) in a 2-way single intersection:
Obs: you need to install stable-baselines3 with ```pip install "stable_baselines3[extra]>=2.0.0a9"``` for [Gymnasium compatibility](https://stable-baselines3.readthedocs.io/en/master/guide/install.html).
```bash
python experiments/dqn_2way-single-intersection.py
```

### Plotting results:
```bash
python outputs/plot.py -f outputs/4x4grid/ppo_conn0_ep2
```
<p align="center">
<img src="outputs/result.png" align="center" width="50%"/>
</p>

## Citing

<!-- start citation -->

Sources used for project:
```bibtex
@misc{sumorl,
    author = {Lucas N. Alegre},
    title = {{SUMO-RL}},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/LucasAlegre/sumo-rl}},
}

@inproceedings{li2018dcrnn_traffic,
  title={Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting},
  author={Li, Yaguang and Yu, Rose and Shahabi, Cyrus and Liu, Yan},
  booktitle={International Conference on Learning Representations (ICLR '18)},
  year={2018}
}
```

<!-- end citation -->


