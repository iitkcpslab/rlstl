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
import time
#env = gym.make('Hopper-v3')
env = gym.make('Humanoid-v3')

#model = SAC.load("Humanoid-v3")
model = SAC.load("sac_humanoid")
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
    px = np.round(info['x_position'],2)
    vx = np.round(info['x_velocity'],2)
    py = np.round(info['y_position'],2)
    vy = np.round(info['y_velocity'],2)
    z,xo,yo,zo,wo =np.round(obs[0:5],2)
    rs1,rs2,re =np.round(obs[16:19],2)
    ls1,ls2,le =np.round(obs[19:22],2)
    #py = info['y_position']
    #vy = info['y_velocity']
    print("id: "+str(i),"   ",px,"    ",py,"     ",z)
    #print("id: "+str(i),"   ",px,"  ",py,"  ",z,"  ",xo," ",yo," ",zo," ",wo)
    #print("id: "+str(i),"   ",px," ",rs1," ",rs2,"  ",re," ",ls1," ",ls2," ",le)
    #print("id: "+str(i)+"   "+str(px)+"  "+str(vx)+" "+str(py)+" "+str(vy))
    #print("info:   "+str(info))
    log = [i,px,z]
    with open('log_humanoid.csv', 'a') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(log)
    #env.render()
    obss.append(obs)
    acts.append(action)
    #if i==2:
    #    time.sleep(20)
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
