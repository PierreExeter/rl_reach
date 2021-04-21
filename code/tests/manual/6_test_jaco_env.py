import gym
import gym_envs

env = gym.make('ReachingJaco-v1')


print("Action space: ", env.action_space)
print(env.action_space.high)
print(env.action_space.low)
print("Observation space: ", env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


env.render()
# env.render(mode="human")  # required by Stable Baselines

for e in range(3):

    obs = env.reset()
    rewards = []

    for i in range(1000):
        print(i)
        # env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        print("action: ", action)
        print("obs: ", obs)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)

        rewards.append(reward)

    cumulative_reward = sum(rewards)
    print("episode {} | cumulative reward : {}".format(e, cumulative_reward))  
    
env.close()