# Load the trained agent
# some_file.py
import sys
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/home/nikhil/RESEARCH/RL/stable-baselines3/')
#sys.path.insert(1, '/home/nikhil/RESEARCH/STL-RESEARCH/rtamt/')
import rtamt


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import pandas as pd
import pickle

import gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import DummyVecEnv
#modelid=4
def compute_cost(modelid):
    #evaluating w.r.t. classical semantics
    #if modelid=="HC":
    #    print(modelid)
    #    exit()
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

    tsc = np.array([]) #total state cost
    tcc = np.array([])  # total control cost
    tsp = np.array([])  # total control cost
    robs = []
    tsteps=np.array([])
    tdist=np.array([])
    tdr=np.array([])
    tseeds=100
    avg_min_rob=[]
    avg_mean_rob=[]
    for sed in range(0,tseeds):
        env.seed(sed)
        sc = np.array([])
        steps = 0
        sp = np.array([])
        cc = np.array([])
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
        obs = env.reset()
        min_rob=1000
        mean_rob=[]
        dr=0
        for i in range(nsteps):
            #action, _states = model.predict(obs, deterministic=True)
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #env.render()

            if modelid=="HalfCheetah-v3":
                x = info['x_position']
                v = info['x_velocity']
                #if x>50:
                #    break
                steps=i
                sc=np.append(sc,50-x)

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('x', 'float')
                            spec.declare_var('v', 'float')
                            spec.spec = 'eventually[0,10](x>0.1)'

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('x', x), ('v', v)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
            elif modelid=="Hopper-v3":
                x = info['x_position']
                #if x>50:
                #    break
                sc=np.append(sc,50-x)
                p = info['x_position']
                v = info['x_velocity']
                z , a = obs[0:2]

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('p', 'float')
                            spec.declare_var('v', 'float')
                            spec.declare_var('z', 'float')
                            spec.declare_var('a', 'float')
                            spec.spec = 'eventually[0,15](p>0.5) and always[0,20]((z>0.7) and (abs(a)<1))'

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
                py = info['y_position']
                x = np.sqrt(np.square([px,py]).sum())
                #if x>50:
                #    break
                sc=np.append(sc,50-x)
                px = info['x_position']
                vx = info['x_velocity']
                z = obs[0]

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('px', 'float')
                            spec.declare_var('vx', 'float')
                            spec.declare_var('z', 'float')
                            spec.spec = 'eventually[0,5](vx>0.2) and always[0,10](abs(z-0.6)<0.4)'

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
                x = info['x_position']
                #if x>50:
                #    break
                sc=np.append(sc,50-x)
                px = info['x_position']
                vx = info['x_velocity']
                z , a = obs[0:2]

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('px', 'float')
                            spec.declare_var('vx', 'float')
                            spec.declare_var('z', 'float')
                            spec.declare_var('a', 'float')
                            spec.spec = 'eventually[0,15] (px>0.5) and always[0,20]((abs(z-0.6)<1.4) and (abs(a)<1))'
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
                x = info['x_position']
                #if x>50:
                #    break
                sc=np.append(sc,50-x)
                px = info['x_position']
                vx = info['x_velocity']
                a1=obs[0]
                a2=obs[1]
                a3=obs[2]

                if i==0:
                            spec = rtamt.STLSpecification()
                            spec.declare_var('px', 'float')
                            spec.declare_var('vx', 'float')
                            spec.declare_var('a1', 'float')
                            spec.declare_var('a2', 'float')
                            spec.declare_var('a3', 'float')
                            spec.spec = 'eventually[0,15](vx>0.1) and always[0,20]((abs(a1)<1) and (abs(a2)<1) and (abs(a3)<1))'

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('px', px), ('vx',vx), ('a1', a1), ('a2', a2), ('a3', a3)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
            elif modelid=="Humanoid-v3":
                x = info['x_position']
                #if x>50:
                #    break
                sc=np.append(sc,50-x)
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
                            spec.spec = 'eventually[0,5](px>0.1) and always[0,10]((abs(py)<2) and (abs(z-1.3)<0.2))'

                            try:
                                spec.parse()
                                spec.pastify()
                            except rtamt.RTAMTException as err:
                                print('RTAMT Exception: {}'.format(err))
                                sys.exit()
                rob = spec.update(i, [('px', px), ('vx', vx), ('py',py), ('z',z)])
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
            if dones==True:
                break

            #sp=np.append(sp,x)
            #cc=np.append(cc,action)
            cc=np.append(cc,np.square(action).sum())
            dr+=rewards
            #robs.append(rob)
            #print(str(sed)+"   "+str(x)+" "+str(rewards))
            #if dones==True:
        tsc=np.append(tsc,np.sum(np.square(sc)))
        tcc=np.append(tcc,np.sum(np.square(cc)))  
        tdr=np.append(tdr,dr)
        tsteps=np.append(tsteps,steps)  
        tdist=np.append(tdist,x)  
        tmp=np.mean(mean_rob)
        avg_min_rob.append(min_rob)
        avg_mean_rob.append(tmp)


    data=np.array(tcc)
    mu=np.mean(data)
    std=np.std(data)
    
    print("###############################################")
    print("#### SUMMMARY : CONTROLLER EVALUATION #########")
    print("###############################################\n")
    print("Control Cost (CC) : ",mu,u"\u00B1",std)

    # data=np.array(tsteps)
    # mu=np.mean(data)
    # std=np.std(data)
    # print("steps mu std var rms:"+str(mu)+"  "+str(std))

    data=np.array(tdist)
    mu=np.mean(data)
    std=np.std(data)
    print("Distance Covered (DC) : ",mu,u"\u00B1",std)

    print("Margin of Satisfaction (MoS) : ",np.mean(avg_mean_rob),u"\u00B1",np.std(avg_mean_rob))
    print("Default Reward (DR) : ",np.mean(tdr),u"\u00B1",np.std(tdr))
    
    #data=np.array(tsc)
    #mu=np.mean(data)
    #std=np.std(data)
    #print("state mu std var rms: "+str(mu)+"  "+str(std))



if __name__ == "__main__":  # noqa: C901
    compute_cost(sys.argv[1])
