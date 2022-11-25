from stable_baselines import DQN

# Deactivate all the DQN extensions to have the original version
# In practice, it is recommend to have them activated
#kwargs = {'double_q': True, 'prioritized_replay': False, 'policy_kwargs': dict(dueling=False)}

#model = A2C('MlpPolicy', 'CartPole-v1', verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")

# Note that the MlpPolicy of DQN is different from the one of PPO
# but stable-baselines handles that automatically if you pass a string
#dqn_model = DQN('MlpPolicy', 'CartPole-v1', verbose=1, **kwargs)
model = DQN('MlpPolicy', 'CartPole-v1',gamma=0.95, learning_rate=0.001, double_q=True, prioritized_replay=False, policy_kwargs=dict(dueling=False), tensorboard_log="./dqn_cartpole_tensorboard/")
#model = DQN('MlpPolicy', 'CartPole-v1', tensorboard_log="./dqn_cartpole_tensorboard/")
model.learn(total_timesteps=int(1e4))

"""
Once the learn function is called, you can monitor the RL agent during or after the training, with the following bash command:

tensorboard --logdir ./a2c_cartpole_tensorboard/
"""

"""
# for more logging use the below custom function

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

model = SAC("MlpPolicy", "Pendulum-v0", tensorboard_log="/tmp/sac/", verbose=1)


class TensorboardCallback(BaseCallback):
    #Custom callback for plotting additional values in tensorboard.

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True


model.learn(50000, callback=TensorboardCallback())

"""

