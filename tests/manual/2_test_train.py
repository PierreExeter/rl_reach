import gym
import widowx_env
import os
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.monitor import Monitor


log_dir = "logs/test/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make('widowx_reacher-v1')
env = Monitor(env, log_dir)

model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=1000, log_interval=10)
model.save(log_dir + "widowx_reach-v1")
