import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import torch
import torch.nn as nn

# Parallel environments
env = make_vec_env("Hopper-v2", n_envs=1)

'''
#model = PPO("MlpPolicy", env, verbose=1)
model = PPO("MlpPolicy", env, tensorboard_log="./ppo_hop_tensorboard", verbose=1)
model.learn(total_timesteps=1e6)
model.save("ppo_hop")

#del model # remove to demonstrate saving and loading

'''
model = PPO.load("ppo_hop")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
