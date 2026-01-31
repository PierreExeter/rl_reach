""" Simple policy evaluation with Stable Baselines """


import gymnasium as gym
import gym_envs
from stable_baselines3 import PPO


env = gym.make('widowx_reacher-v1', render_mode="human")
# model = PPO(MlpPolicy, env, verbose=1)
model = PPO.load("logs/test/widowx_reach-v1")

obs, info = env.reset()

for t in range(3000):
    print(t)
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
