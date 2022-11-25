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
from rtamt.spec.stl.discrete_time.specification import Semantics


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



def monitor(modelid):
    #checking w.r.t. classical semantics
    open('gvars.py', 'w').close()
    file1 = open('gvars.py', 'w')
    s="rob_type=1"
    file1.write(s)
    file1.close()

    if modelid==1:
        env = gym.make('Pendulum-v0') 
    elif modelid==2:
        env = gym.make('MountainCarContinuous-v0') 
    elif modelid==3:
        env = gym.make('HalfCheetah-v3') 
    elif modelid==4:
        env = gym.make('Hopper-v3') 
    elif modelid==5:
        env = gym.make('Ant-v3') 
    elif modelid==6:
        env = gym.make('ReacherBulletEnv-v0') 
    elif modelid==7:
        env = gym.make('Walker2d-v3') 
    elif modelid==8:
        env = gym.make('Swimmer-v3') 
    elif modelid==9:
        env = gym.make('Humanoid-v3') 


    robs = []
    tseeds=100
    sat=0
    for sed in range(0,tseeds):
        env.seed(sed)
        if modelid==1:
            model = SAC.load("sac_pend")
            nsteps=200 
        elif modelid==2:
            model = SAC.load("sac_mc")
            nsteps=999 
        elif modelid==3:
            model = SAC.load("sac_hc")
            nsteps=1000 
        elif modelid==4:
            model = SAC.load("sac_hop")
            nsteps=1000 
        elif modelid==5:
            model = SAC.load("sac_ant")
            nsteps=1000 
        elif modelid==6:
            model = SAC.load("sac_reacher")
            nsteps=150
        elif modelid==7:
            model = SAC.load("sac_walker")
            nsteps=1000
        elif modelid==8:
            model = SAC.load("sac_swimmer")
            nsteps=1000
        elif modelid==9:
            model = SAC.load("sac_humanoid")
            nsteps=1000
        #env.render()

        #############

        # Enjoy trained agent
        data = {}
        data['p'] = []
        data['z'] = []
        data['a'] = []
        obs = env.reset()
        for i in range(nsteps):
            #action, _states = model.predict(obs, deterministic=True)
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #env.render()

            p = info['x_position']
            z , a =obs[0:2]
            u = np.clip(action, -2, 2)[0] 
            data['p'].append((i,p))
            data['z'].append((i,z))
            data['a'].append((i,a))
        # print(data)
        # print(data['t'][0])
        # print(data['t'][1])
        # print(data['t'][2])

        spec = rtamt.STLDiscreteTimeSpecification(1)
        spec.name = 'Comparison'
        spec.declare_var('p', 'float')
        spec.declare_var('z', 'float')
        spec.declare_var('a', 'float')

        #spec.spec = 'always[0,15] (eventually[0,10](p>1) and (z>0.7) and (abs(a)<1))'  
        spec.spec = 'always[0,999] ((z>0.7) and (abs(a)<1))'  

        spec.semantics = Semantics.STANDARD
        try:
            spec.parse()
            spec.pastify()
        except rtamt.STLParseException as err:
            print('STL Parse Exception: {}'.format(err))
            sys.exit()

        for i in range(len(data['p'])):
            rob = spec.update(i, [('p', data['p'][i][1]), ('z', data['z'][i][1]), ('a', data['a'][i][1])])
            #print(str(data['t'][i][1])+"  "+str(data['td'][i][1])+"  "+str(rob))
        
        #print(rob) 
        if rob>=0:
            sat+=1
    print("Experiment ",sat,"/",tseeds," : safety satisfaction")
        


if __name__ == "__main__":  # noqa: C901
    monitor(int(sys.argv[1]))



