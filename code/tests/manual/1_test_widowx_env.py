""" Test env with random actions """

import time
import gymnasium as gym
import gym_envs
from stable_baselines3.common.env_checker import check_env


env = gym.make('widowx_reacher-v49', render_mode="human")
print("any warnings?", check_env(env))

# Comment this out for goal environments
print("Action space: ", env.action_space)
print("low: ", env.action_space.low)
print("high: ", env.action_space.high)
print("Observation space: ", env.observation_space)
print("low: ", env.observation_space.low)
print("high: ", env.observation_space.high)

env.render()

for episode in range(5):
    obs, info = env.reset()
    rewards = []

    for t in range(100):
        action = env.action_space.sample()
        # action = [2, 0, 0, 0, 0, 0]
        # action[0] = 0
        # action[1] = 0
        # action[2] = 0
        # action[3] = 0
        # action[4] = 0
        # action[5] = 0

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # env.render()

        print("action: ", action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)
        print("timestep: ", t)

        rewards.append(reward)
        time.sleep(1. / 30.)
        # time.sleep(3)

    cumulative_reward = sum(rewards)
    print(
        "episode {} | cumulative reward : {}".format(
            episode,
            cumulative_reward))

env.close()
