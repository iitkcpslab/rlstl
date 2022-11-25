import gym
import sys
#sys.path.insert(1, '/home/nikhil/RESEARCH/RL/SSFC/')
import argparse
import warnings
warnings.filterwarnings("ignore")


import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import timeit
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
#print(sys.argv[1])
#exit()

def run_exp(env_id,rob_type,run):


    open('gvars.py', 'w').close()
    file1 = open('gvars.py', 'w')
    if rob_type==1:
        name="cls"
        s="rob_type=1"
    elif rob_type==2:
        name="agm"
        s="rob_type=2"
    elif rob_type==3:
        name="smax"
        s="rob_type=3"
    elif rob_type==4:
        name="lse"
        s="rob_type=4"
    elif rob_type==5:
        name="sss"
        s="rob_type=5"

    file1.write(s)
    file1.close()
    print(env_id)


    if env_id=="HalfCheetah-v3":  
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = DummyVecEnv([lambda: gym.make('HalfCheetah-v3')])

        start = timeit.default_timer()
        model = SAC('MlpPolicy','HalfCheetah-v3', learning_starts=10000, use_sde=False,  tensorboard_log="./sac_hc_tensorboard", verbose=1, seed=594371)
        #model.learn(total_timesteps=int(1.5e6),reward_type="STL",sem=name)
        model.learn(total_timesteps=int(1e6),reward_type="STL",sem=name)
        #model.learn(total_timesteps=int(2e6),reward_type="STL",sem=name)
        model.save("sac_HalfCheetah-v3_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("hc.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()


    elif env_id=="Hopper-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        #env = gym.make('Hopper-v3')
        env = DummyVecEnv([lambda: gym.make('Hopper-v3')])

        start = timeit.default_timer()
        model = SAC('MlpPolicy','Hopper-v3',learning_rate=0.0003, buffer_size=1000000, learning_starts=10000, use_sde=False,  tensorboard_log="./sac_hop_tensorboard", verbose=1, seed=594371)
        #model.learn(total_timesteps=int(1e6),reward_type="STL")
        model.learn(total_timesteps=int(1e6),reward_type="STL",sem=name)
        model.save("sac_Hopper-v3_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("hopper.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()


    elif env_id=="Walker2d-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = DummyVecEnv([lambda: gym.make('Walker2d-v3')])

        #env = VecNormalize(env, norm_obs=True)
        #model = SAC('MlpPolicy','Hopper-v3', verbose=1, tensorboard_log="./sac_hop_tensorboard/")
        model = SAC('MlpPolicy','Walker2d-v3',learning_starts=10000,  use_sde=False,  tensorboard_log="./sac_walker_tensorboard", verbose=1, seed=594371)
        #model.learn(total_timesteps=int(1e6),reward_type="STL")
        model.learn(total_timesteps=int(1e6),reward_type="STL",sem=name)

        model.save("sac_Walker2d-v3_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("walker.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
        


    elif env_id=="Ant-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = gym.make('Ant-v3')

        #env = DummyVecEnv([lambda: gym.make('ReacherBulletEnv-v0')])
        #env = VecNormalize(env, norm_obs=True)
        model = SAC('MlpPolicy','Ant-v3',learning_starts=10000, use_sde=False,  tensorboard_log="./sac_ant_tensorboard", verbose=1, seed=594371)
        model.learn(total_timesteps=int(1e6),reward_type="STL",sem=name)

        model.save("sac_Ant-v3_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))  
        text_file = open("ant.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
    

    elif env_id=="Swimmer-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = DummyVecEnv([lambda: gym.make('Swimmer-v3')])

        #env = VecNormalize(env, norm_obs=True)
        #model = SAC('MlpPolicy','Hopper-v3', verbose=1, tensorboard_log="./sac_hop_tensorboard/")
        model = SAC('MlpPolicy','Swimmer-v3',learning_starts=10000, gamma=0.9999, use_sde=False,  tensorboard_log="./sac_swimmer_tensorboard", verbose=1, seed=594371)
        model.learn(total_timesteps=int(1.5e6),reward_type="STL",sem=name)
        #model.learn(total_timesteps=int(1e6),reward_type="STL",sem=name)
        #model.learn(total_timesteps=int(2e6),reward_type="STL",sem=name)
        model.save("sac_Swimmer-v3_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("swimmer.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
    
    elif env_id=="Humanoid-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env = DummyVecEnv([lambda: gym.make('Humanoid-v3')])

        #env = VecNormalize(env, norm_obs=True)
        model = SAC('MlpPolicy','Humanoid-v3',learning_starts=10000, tensorboard_log="./sac_humanoid_tensorboard", verbose=1, seed=594371)
        model.learn(total_timesteps=int(2e6),reward_type="STL",sem=name)
        #model.learn(total_timesteps=int(3e6),reward_type="STL")

        model.save("sac_Humanoid-v3_"+name+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("humanoid.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()
    


if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="Hopper-v3")
    parser.add_argument("--sem", help="STL semantics", default=1, type=int, required=False)
    parser.add_argument("--run", help="run id", default=10, type=int, required=False)
  
    args = parser.parse_args()

    env_id = args.env
    sem = args.sem
    run = args.run
    run_exp(env_id,sem,run)
