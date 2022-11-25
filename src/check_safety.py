# Load the trained agent
# some_file.py
import sys
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/home/nikhil/RESEARCH/RL/stable-baselines3/')
#sys.path.insert(1, '/home/nikhil/RESEARCH/RL/stable-baselines/')
#sys.path.insert(1, '/home/nikhil/RESEARCH/STL-RESEARCH/rtamt/')
import rtamt
from rtamt.spec.stl.discrete_time.specification import Semantics


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import pickle

import gym
#from stable_baselines import SAC
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import DummyVecEnv



def monitor(modelid):
    #checking w.r.t. classical semantics
    open('gvars.py', 'w').close()
    file1 = open('gvars.py', 'w')
    s="rob_type=1"
    file1.write(s)
    file1.close()

    if modelid=="HalfCheetah-v3":
        env = gym.make('HalfCheetah-v3') 
    elif modelid=="Hopper-v3":
        env = gym.make('Hopper-v3') 
    elif modelid=="Ant-v3":
        env = gym.make('Ant-v3') 
    elif modelid=="Walker2d-v3":
        env = gym.make('Walker2d-v3') 
    elif modelid=="Swimmer-v3":
        env = gym.make('Swimmer-v3') 
    elif modelid=="Humanoid-v3":
        env = gym.make('Humanoid-v3') 


    robs = []
    tseeds=100
    sat=0
    for sed in range(0,tseeds):
        env.seed(sed)
        if modelid=="HalfCheetah-v3":
            model = SAC.load("sac_HalfCheetah-v3")
            nsteps=1000 
        elif modelid=="Hopper-v3":
            model = SAC.load("sac_Hopper-v3")
            nsteps=1000 
        elif modelid=="Ant-v3":
            model = SAC.load("sac_Ant-v3")
            nsteps=1000 
        elif modelid=="Walker2d-v3":
            model = SAC.load("sac_Walker2d-v3")
            nsteps=1000
        elif modelid=="Swimmer-v3":
            model = SAC.load("sac_Swimmer-v3")
            nsteps=1000
        elif modelid=="Humanoid-v3":
            model = SAC.load("sac_Humanoid-v3")
            nsteps=1000
        #env.render()

        #############

        # Enjoy trained agent
        data = {}
        data['p'] = []
        data['z'] = []
        data['a'] = []
        min_rob=1000
        mean_rob=[]
        obs = env.reset()
        for i in range(nsteps):
            #action, _states = model.predict(obs, deterministic=True)
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #env.render()

            if modelid=="HalfCheetah-v3":
                print("No safety requirement for HalfCheetah-v3")
                exit()
            elif modelid=="Hopper-v3":
                p = info['x_position']
                v = info['x_velocity']
                z , a =obs[0:2]
                u = np.clip(action, -2, 2)[0] 
                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('p', 'float')
                            spec.declare_var('v', 'float')
                            spec.declare_var('z', 'float')
                            spec.declare_var('a', 'float')
                            spec.spec = 'always[0,20] ((z>0.7) and (abs(a)<1))'  

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('p', p), ('v', v), ('z', z), ('a', a)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
            elif modelid=="Ant-v3":
                px = info['x_position']
                vx = info['x_velocity']
                z = obs[0]

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('px', 'float')
                            spec.declare_var('vx', 'float')
                            spec.declare_var('z', 'float')
                            spec.spec = 'always[0,10](abs(z-0.6)<0.4)'

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('px', px), ('vx', vx), ('z', z)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
            elif modelid=="Walker2d-v3":
                px = info['x_position']
                vx = info['x_velocity']
                z , a = obs[0:2]

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('px', 'float')
                            spec.declare_var('vx', 'float')
                            spec.declare_var('z', 'float')
                            spec.declare_var('a', 'float')
                            spec.spec = 'always[0,20]((abs(z-0.6)<1.4) and (abs(a)<1))' 

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('px', px), ('vx', vx), ('z', z), ('a', a)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
            elif modelid=="Swimmer-v3":
                px = info['x_position']
                vx = info['x_velocity']
                a1=obs[0]
                a2=obs[1]
                a3=obs[2]

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('px', 'float')
                            spec.declare_var('a1', 'float')
                            spec.declare_var('a2', 'float')
                            spec.declare_var('a3', 'float')
                            spec.spec = 'always[0,20]((abs(a1)<1) and (abs(a2)<1) and (abs(a3)<1))'

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('px', px), ('a1', a1), ('a2', a2), ('a3', a3)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
            elif modelid=="Humanoid-v3":
                px = info['x_position']
                py = info['y_position']
                vx = info['x_velocity']
                z = obs[0]

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('px', 'float')
                            spec.declare_var('py', 'float')
                            spec.declare_var('vx', 'float')
                            spec.declare_var('z', 'float')
                            spec.spec = 'always[0,10]((abs(py)<2) and (abs(z-1.3)<0.2))'

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('px', px), ('vx', vx), ('py',py), ('z',z)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)

        #print(min_rob) 
        if min_rob>=0:
            sat+=1
        #exit()
    print("Safety Satisfaction (SAT) : ",sat,"/",tseeds)
        


if __name__ == "__main__":  # noqa: C901
    monitor(sys.argv[1])



