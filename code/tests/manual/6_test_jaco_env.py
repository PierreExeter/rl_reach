import gymnasium as gym
import gym_envs
from stable_baselines3.common.env_checker import check_env

env = gym.make('ReachingJaco-v10')
# print("any warnings?", check_env(env))


print("Action space: ", env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print("Observation space: ", env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

env.render()
# env.render(mode="human")  # required by Stable Baselines


for e in range(3):

    obs, info = env.reset()
    rewards = []

    for i in range(100):
        print(i)
        # env.render()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print("action: ", action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)

        rewards.append(reward)

    cumulative_reward = sum(rewards)
    print("episode {} | cumulative reward : {}".format(e, cumulative_reward))  
    
env.close()