""" Test env with random actions """

import time
import gym
import widowx_env
from stable_baselines3.common.env_checker import check_env


env = gym.make('widowx_reacher-v37')
print("any warnings?", check_env(env))

# Comment this out for goal environments
print("Action space: ", env.action_space)
print("low: ", env.action_space.low)
print("high: ", env.action_space.high)
print("Observation space: ", env.observation_space)
print("low: ", env.observation_space.low)
print("high: ", env.observation_space.high)

# env.render()
frame = env.render()

for episode in range(5):
    obs = env.reset()
    rewards = []

    for t in range(100):
        action = env.action_space.sample()
        # action = [1, 0, 0, 0, 0, 0]

        obs, reward, done, info = env.step(action)
        frame = env.render()

        print("action: ", action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)
        print("timestep: ", t)

        rewards.append(reward)
        time.sleep(1. / 30.)

    cumulative_reward = sum(rewards)
    print(
        "episode {} | cumulative reward : {}".format(
            episode,
            cumulative_reward))

env.close()
