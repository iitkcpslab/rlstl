
# Load the trained agent
# some_file.py
import sys
import numpy as np
import pickle
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/nikhil/RESEARCH/RL/stable-baselines3/')
sys.path.insert(1, '/home/nikhil/RESEARCH/RL/stable-baselines/')
sys.path.insert(1, '/home/nikhil/RESEARCH/STL-RESEARCH/rtamt/')
import rtamt


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import pickle

from stable_baselines.ddpg import DDPG
from stable_baselines import DQN
from stable_baselines import PPO2
from stable_baselines3 import SAC
from stable_baselines.common.vec_env import DummyVecEnv
import gym
import pybullet_envs
#from stable_baselines import SAC
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env 
from stable_baselines.common.vec_env import DummyVecEnv
modelid=4
if modelid==1:
        env = gym.make('Pendulum-v0') 
        model = SAC.load("sac_pend")
        nsteps=200 
elif modelid==2:
        env = gym.make('MountainCarContinuous-v0') 
        model = SAC.load("sac_mc")
        nsteps=999 
elif modelid==3:
        env = gym.make('HalfCheetah-v3') 
        model = SAC.load("sac_hc")
        nsteps=1000 
elif modelid==4:
        env = gym.make('Hopper-v3') 
        model = SAC.load("sac_hop")
        modelr = SAC.load("sac_hopr")
        nsteps=1000 
elif modelid==5:
        env = gym.make('Ant-v3') 
        model = SAC.load("sac_ant")
        nsteps=1000 
elif modelid==6:
        env = gym.make('ReacherBulletEnv-v0') 
        model = SAC.load("sac_reacher")
        nsteps=150
elif modelid==7:
        env = gym.make('Walker2d-v3') 
        model = SAC.load("sac_walker")
        nsteps=1000
elif modelid==8:
        env = gym.make('Swimmer-v3') 
        model = SAC.load("sac_swimmer")
        nsteps=1000
elif modelid==9:
        env = gym.make('Humanoid-v3') 
        model = SAC.load("sac_humanoid")
        nsteps=1000

robs = []
# box =[P(green),P(blue),P(red),P(yellow)]
box_1 = [0.25, 0.33, 0.23, 0.19]
box_2 = [0.21, 0.21, 0.32, 0.26]

import numpy as np
from scipy.special import rel_entr

def kl_divergence(a, b):
    ret = 0
    for i in range(len(a)):
        #print(a[i]," ",b[i])
        if a[i]==0:
            a[i]=0.0001
        if b[i]==0:
            b[i]=0.0001

        ret += a[i]*np.log(a[i]/b[i])
    return ret


def normalise(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

tseeds=100
res = np.array([])
for sed in range(0,tseeds):
    act1=np.empty([1,3])
    act2=np.empty([1,3])
    env.seed(sed)
    #env.render()

    # Enjoy trained agent
    obs = env.reset()
    for i in range(nsteps):
        #action, _states = model.predict(obs, deterministic=True)
        a1, s1 = model.predict(obs)
        a2, s2 = modelr.predict(obs)
        act1=np.vstack([act1,a1])
        act2=np.vstack([act2,a2])
        #env.render()
        #print(a1," ",a2)
    #print(act1)
    for i in range(len(act1)):
        act1[i]=(act1[i] - np.min(act1[i]))/(np.max(act1[i]) - np.min(act1[i])+0.00001)
        act2[i]=(act2[i] - np.min(act2[i]))/(np.max(act2[i]) - np.min(act2[i])+0.00001)

    sum1 = np.sum(act1, axis=1).reshape(nsteps+1,1)
    sum2 = np.sum(act2, axis=1).reshape(nsteps+1,1)
    #print(sum1," ",sum2)
    #print(act1)
    #print(act2)
    #print(len(act1)
    p=act1/(sum1+0.0001)
    q=act2/(sum2+0.0001)
    kld=[]
    for i in range(len(p[0])):
        kld.append(kl_divergence(p[:,i],q[:,i]))
    kld=np.min(kld)
    res=np.append(res,kld)
    print(kld)
    #kld=sum(rel_entr(act1,act2))
    #print(kld)
print(np.mean(res))


