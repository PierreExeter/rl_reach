""" Simple training with Stable Baselines """


import os
import gym
import widowx_env
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor


LOG_DIR = "logs/test/"
os.makedirs(LOG_DIR, exist_ok=True)

env = gym.make('widowx_reacher-v1')
env = Monitor(env, LOG_DIR)

model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000, log_interval=10)
model.save(LOG_DIR + "widowx_reach-v1")
