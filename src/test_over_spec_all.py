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
#modelid=5
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

    tsc = np.array([]) #total state cost
    tcc = np.array([])  # total control cost
    tsp = np.array([])  # total control cost
    robs = []
    tseeds=100
    avg_min_rob=[]
    avg_mean_rob=[]
    for sed in range(0,tseeds):
        env.seed(sed)
        sc = np.array([])
        sp = np.array([])
        cc = np.array([])
        #env = gym.make('BipedalWalker-v3') 
        #env = gym.make('CartPole-v1') 
        #env = gym.make('LunarLanderContinuous-v2') 
        #env = gym.make('HalfCheetah-v3') 
        #env = DummyVecEnv([lambda:env])
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
        #model = PPO2.load("ddpg_bpw")
        #env.render()

        #############

        # Enjoy trained agent
        obs = env.reset()
        min_rob=1000
        mean_rob=[]
        for i in range(nsteps):
            #action, _states = model.predict(obs, deterministic=True)
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #env.render()
            if modelid==3:
                x = info['x_position']
                v = info['x_velocity']

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('x', 'float')
                            spec.declare_var('v', 'float')
                            #functional specification - for training
                            #spec.spec = 'always[0,1000] (p>0.8)'
                            #spec.spec = 'eventually[0,10] ((abs(x)<0.01) and (abs(y)<0.01))'
                            #demospec
                            spec.spec = 'always[0,10] ((eventually[0,10] (x>1)) and (v>0.2))'

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('x', x), ('v', v)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)

            elif modelid==4:
                p = info['x_position']
                v = info['x_velocity']
                z , a = obs[0:2]

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('p', 'float')
                            spec.declare_var('v', 'float')
                            spec.declare_var('z', 'float')
                            spec.declare_var('a', 'float')
                            #functional specification - for training
                            #spec.spec = 'eventually[0,10](p>0.1) and always[0,15]((z>0.7) and (abs(a)<1))'
                            spec.spec = 'eventually[0,10](p>0.1)'

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('p', p), ('v', v), ('z', z), ('a', a)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
            elif modelid==5:
                px = info['x_position']
                vx = info['x_velocity']
                z = obs[0]

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('px', 'float')
                            spec.declare_var('vx', 'float')
                            spec.declare_var('z', 'float')
                            #functional specification - for training
                            #spec.spec = 'always[0,1000] (p>0.8)'
                            #spec.spec = 'eventually[0,10] ((abs(x)<0.01) and (abs(y)<0.01))'
                            #demospec
                            spec.spec = 'always[0,10](eventually[0,5](vx>0.01) and (abs(z-0.6)<0.4))'

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('px', px), ('vx', vx), ('z', z)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
            elif modelid==7:
                px = info['x_position']
                vx = info['x_velocity']
                z , a = obs[0:2]

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('px', 'float')
                            spec.declare_var('vx', 'float')
                            spec.declare_var('z', 'float')
                            spec.declare_var('a', 'float')
                            #functional specification - for training
                            #spec.spec = 'always[0,1000] (p>0.8)'
                            #spec.spec = 'eventually[0,10] ((abs(x)<0.01) and (abs(y)<0.01))'
                            #demospec
                            spec.spec = 'always[0,10](eventually[0,5] (px>1) and (abs(z-0.6)<1.4) and (abs(a)<1))' 

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('px', px), ('vx', vx), ('z', z), ('a', a)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
            elif modelid==8:
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
                            #functional specification - for training
                            #spec.spec = 'always[0,1000] (p>0.8)'
                            #spec.spec = 'eventually[0,10] ((abs(x)<0.01) and (abs(y)<0.01))'
                            #demospec
                            spec.spec = 'always[0,10](eventually[0,10](px>1) and (abs(a1)<1) and (abs(a2)<1) and (abs(a3)<1))'

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('px', px), ('a1', a1), ('a2', a2), ('a3', a3)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
            elif modelid==9:
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
                            #functional specification - for training
                            #spec.spec = 'always[0,1000] (p>0.8)'
                            #spec.spec = 'eventually[0,10] ((abs(x)<0.01) and (abs(y)<0.01))'
                            #demospec
                            spec.spec = 'always[0,10](eventually[0,5] (px>0) and (abs(py)<2) and (abs(z-1)<1))' 

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('px', px), ('vx', vx), ('py',py), ('z',z)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
        tmp=np.mean(mean_rob)
        avg_min_rob.append(min_rob)
        avg_mean_rob.append(tmp)


    print(np.mean(avg_min_rob)," ",np.std(avg_min_rob))
    print(np.mean(avg_mean_rob)," ",np.std(avg_mean_rob))


if __name__ == "__main__":  # noqa: C901
    monitor(int(sys.argv[1]))
