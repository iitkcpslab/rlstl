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
import time
import csv
#env = gym.make('Swimmer-v3')
env = gym.make('HalfCheetah-v3')

#model = SAC.load("Swimmer-v3")
#model = SAC.load("Ant-v3")
model = SAC.load("sac_hc")
#env.render()

env.seed(0)
####
#############

# Enjoy trained agent
#env.render()
obs = env.reset()
obss = []
acts = []
robs = []
ctr=0

for i in range(1000):
    #action, _states = model.predict(obs, deterministic=True)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    x = info['x_position']
    v = info['x_velocity']
    print("id: "+str(i)+"   "+str(x)+"  "+str(v)+" "+str(rewards))
    #print("info:   "+str(info))
    log = [i,x,v]
    with open('log_hc.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(log)
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
