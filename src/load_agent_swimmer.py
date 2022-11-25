# Load the trained agent
# some_file.py
import sys
import numpy as np
import pickle
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/nikhil/RESEARCH/RL/stable-baselines3/')
sys.path.insert(1, '/home/nikhil/RESEARCH/STL-RESEARCH/rtamt/')
import gym
import pybullet_envs
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import csv

env = gym.make('Swimmer-v3')

model = SAC.load("sac_swimmer")
#model = SAC.load("Swimmer-v3")
#env.render()

#env.seed(0)
####
#############

# Enjoy trained agent
obs = env.reset()
obss = []
acts = []
robs = []
ctr=0
for i in range(1000):
    #action, _states = model.predict(obs, deterministic=True)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    px = info['x_position']
    vx = info['x_velocity']
    py = info['y_position']
    vy = info['y_velocity']
    a1, a2, a3 =obs[0:3]
    #py = info['y_position']
    #vy = info['y_velocity']
    print("id: "+str(i)+"   "+str(px)+"   ",a1,"   ",a2,"   ",a3)
    #print("info:   "+str(info))
    #env.render()
    obss.append(obs)
    acts.append(action)
    #robs.append(rob)
    #print(str(obs)+"   "+str(rewards))
    #if abs(info['x_position'])>50:
    #    break
    if dones==True:
        break

with open("odata.pkl", 'wb') as file:
    pickle.dump(obss, file)
with open("udata.pkl", 'wb') as file:
    pickle.dump(acts, file)
'''
with open("rdata.pkl", 'wb') as file:
    pickle.dump(robs, file)
'''
