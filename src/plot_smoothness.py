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
#modelid=4

def monitor(modelid):
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

    sv = np.array([]) # state values
    cv = np.array([])  # control values
    robs = []
    tseeds=1
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
        #model = PPO2.load("ddpg_bpw")
        #env.render()

        #############

        # Enjoy trained agent
        obs = env.reset()
        for i in range(nsteps):
            #action, _states = model.predict(obs, deterministic=True)
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #env.render()

            if modelid==1:
                sv=np.append(sv,np.arccos(obs[0]))
            elif modelid==2:
                sv=np.append(sv,obs[0])
            elif modelid==3:
                x = info['x_position']
                sv=np.append(sv,x)
            elif modelid==4:
                x = info['x_position']
                sv=np.append(sv,x)
            elif modelid==5:
                px = info['x_position']
                py = info['y_position']
                x = np.sqrt(np.square([px,py]).sum())
                sv=np.append(sv,px)
            elif modelid==6:
                px = obs[2]
                py = obs[3]
                x = np.square([px,py]).sum()
                sv=np.append(sv,x)
            elif modelid==7:
                x = info['x_position']
                sv=np.append(sv,x)
            elif modelid==8:
                x = info['x_position']
                sv=np.append(sv,x)
            elif modelid==9:
                x = info['x_position']
                sv=np.append(sv,x)

            #sp=np.append(sp,x)
            #cc=np.append(cc,action)
            cv=np.append(cv,np.square(action).sum())
            #robs.append(rob)
            #print(str(sed)+"   "+str(x)+" "+str(rewards))
            #if dones==True:
        


    feedback_list = []
    time_list = []
    outputs = []
    setpoint_list = []



    # time_sm = np.array(time_list)
    # time_smooth = np.linspace(time_sm.min(), time_sm.max(), 300)

    # feedback_smooth = spline(time_list, feedback_list, time_smooth)
    # Using make_interp_spline to create BSpline
    # helper_x3 = make_interp_spline(time_list, feedback_list)
    # feedback_smooth = helper_x3(time_smooth)

    f,(ax1,ax2) = plt.subplots(2,1,sharex=True, squeeze=True)
    ax1.set_title('Plot Smoothness')
    ax1.plot(sv, label='Internal State')
    #ax1.plot(time_list, setpoint_list, label='Set-point')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('State')
    ax1.grid(True)
    ax1.legend()

    ax2.plot(cv, label='Actions taken')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('Action')
    ax2.grid(True)
    ax2.legend()

    plt.show()


if __name__ == "__main__":  # noqa: C901
    monitor(int(sys.argv[1]))