import gym
#sys.path.insert(1, '/home/nikhil/RESEARCH/RL/stable-baselines3/')
#sys.path.insert(1, '/home/nikhil/RESEARCH/STL-RESEARCH/rtamt/')
import pybullet_envs
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3 import SAC

env_id = 'Swimmer-v3'
#env_id = 'Walker2d-v3'
#env_id = 'Humanoid-v3'
#env_id = 'BipedalWalker-v3'
#env_id = 'HalfCheetah-v3'
#env_id = 'Hopper-v3'
#env_id = 'Ant-v3'
video_folder = 'videos/'
video_length = 1000

env = DummyVecEnv([lambda: gym.make(env_id)])
model = SAC.load("sac_swimmer")
#model = SAC.load("sac_walker")
#model = SAC.load("sac_bpwalker")
#model = SAC.load("sac_hc")
#model = SAC.load("sac_humanoid")
#model = SAC.load("sac_hop")
#model = SAC.load("sac_ant")


obs = env.reset()

# Record the video starting at the first step
env = VecVideoRecorder(env, video_folder,
                       record_video_trigger=lambda x: x == 0, video_length=video_length,
                       name_prefix=f"random-agent-{env_id}")

env.seed(0)
env.reset()
for _ in range(video_length + 1):
    #action = [env.action_space.sample()]
    action, _states = model.predict(obs)
    obs, _, _, _ = env.step(action)
# Save the video
env.close()
