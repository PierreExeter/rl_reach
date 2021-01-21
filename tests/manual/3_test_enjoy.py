""" Simple policy evaluation with Stable Baselines """


import gym
import widowx_env
from stable_baselines3 import PPO


env = gym.make('widowx_reacher-v1')
# model = PPO(MlpPolicy, env, verbose=1)
model = PPO.load("logs/test/widowx_reach-v1")

obs = env.reset()
env.render(mode="human")

for t in range(3000):
    print(t)
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

env.close()
