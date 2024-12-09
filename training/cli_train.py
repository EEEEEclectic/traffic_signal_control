import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser()

    # General experiment settings
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Directory to store training results, logs, and models")
    parser.add_argument("--exp_name", type=str, default="default_exp",
                        help="Name of the experiment (directory will be created inside out_dir)")

    # Environment settings
    parser.add_argument("--net_file", type=str, default="sumo_rl/nets/RESCO/grid4x4/grid4x4.net.xml")
    parser.add_argument("--route_file", type=str, default="sumo_rl/nets/RESCO/grid4x4/grid4x4_1.rou.xml")
    parser.add_argument("--num_seconds", type=int, default=60, help="Simulation length in seconds")
    parser.add_argument("--begin_time", type=int, default=100, help="Begin time in SUMO simulation")

    # RL settings
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")

    # Model type and approach
    parser.add_argument("--model_type", type=str, choices=["dcrnn", "transformer"], default="dcrnn",
                        help="Which model architecture to use")
    parser.add_argument("--approach", type=str, choices=["centralized", "decentralized"], default="centralized",
                        help="Centralized (single-agent) or decentralized (multi-agent) training")

    # Model hyperparameters
    parser.add_argument("--num_transformer_layers", type=int, default=2, help="Number of transformer layer")
    parser.add_argument("--num_proj_layers", type=int, default=2, help="Number of linear projection layer")
    parser.add_argument("--hidden_features", type=int, default=128, help="Number of hidden feature")

    # Graph hyperparameters
    parser.add_argument("--k", type=int, default=2, help="Number of last k")

    # Additional arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cuda or cpu")
    parser.add_argument("--fixed_ts", action="store_true", help="If set, no RL control and fixed TS is used")
    parser.add_argument("--reward_fn", type=str, default="weighted_wait_queue", help="Reward function name")

    args = parser.parse_args()
    return args


def save_config(args):
    exp_path = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(exp_path, exist_ok=True)
    with open(os.path.join(exp_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
