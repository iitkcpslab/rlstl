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

env = gym.make('Hopper-v3')
#env = gym.make('Walker2d-v3')

#model = SAC.load("sac_walker")
model = SAC.load("sac_hop")
#env.render()

env.seed(0)
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
    z , a =obs[0:2]
    #py = info['y_position']
    #vy = info['y_velocity']
    print("id: "+str(i)+"   "+str(px)+"   "+str(z-0.7)+"    "+str(1-abs(a)))
    #print("id: "+str(i)+"   "+str(px)+"  "+str(vx)+" "+str(py)+" "+str(vy))
    #print("info:   "+str(info))
    log = [i,px,z,a]
    with open('log_hop.csv', 'a') as csvfile:
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
