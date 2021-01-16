import gym
import time
import widowx_env
from stable_baselines3.common.env_checker import check_env


env = gym.make('widowx_reacher-v12')
print("any warnings?", check_env(env))

# Comment this for goal environments
# print("Action space: ", env.action_space)
# print(env.action_space.high)
# print(env.action_space.low)
# print("Observation space: ", env.observation_space)
# print(env.observation_space.high)
# print(env.observation_space.low)


# env.render()


for episode in range(2):
    obs = env.reset()
    rewards = []

    for t in range(100):
        action = env.action_space.sample()

        obs, reward, done, info = env.step(action)

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
