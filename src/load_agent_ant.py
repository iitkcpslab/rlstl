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
#env = gym.make('Walker2d-v3')
env = gym.make('Ant-v3')

#model = SAC.load("sac_walker")
#model = SAC.load("Ant-v3")
model = SAC.load("sac_ant")
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
    py = info['y_position']
    vx = info['x_velocity']
    z = obs[0]
    #py = info['y_position']
    #vy = info['y_velocity']
    #print("id: "+str(i)+"   "+str(vx)+"  "+str(z))
    print("id: "+str(i)+"   "+str(px)+"   "+str(py),"  ",z)
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
