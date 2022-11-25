import argparse
import os
import sys
sys.path.insert(1, '/home/nikhil/RESEARCH/RL/stable-baselines3/')
sys.path.insert(1, '/home/nikhil/RESEARCH/STL-RESEARCH/rtamt/')
import gym

import numpy as np
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecVideoRecorder


if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="CartPole-v1")
    parser.add_argument("-f", "--model", help="location of file", type=str, default="")
    parser.add_argument("-o", "--output-folder", help="Output folder", type=str)
    parser.add_argument("--algo", help="RL Algorithm", default="ppo", type=str, required=False)
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument("--deterministic", action="store_true", default=False, help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_true", default=False, help="Use stochastic actions")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument(
        "--no-render", action="store_true", default=False, help="Do not render the environment (useful for tests)"
    )
    parser.add_argument("--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    args = parser.parse_args()

    env_id = args.env
    algo = args.algo
    video_folder = args.output_folder
    seed = args.seed
    video_length = args.n_timesteps
    n_envs = args.n_envs
    model=args.model
    

    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]

    env = gym.make('HalfCheetah-v3') 
    
    model = SAC.load("sac_hc")
    

    if video_folder is None:
        video_folder = "videos"

    # Note: apparently it renders by default
    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="sac_hc",
    )

    obs = env.reset()
    try:
        for _ in range(100):
            action, _states = model.predict(obs)
            obs, _, dones, _ = env.step(action)
            episode_starts = dones
            if not args.no_render:
                env.render()
    except KeyboardInterrupt:
        pass

    env.close()
