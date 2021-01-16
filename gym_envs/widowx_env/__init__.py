from gym.envs.registration import register


# Pybullet environment + fixed goal + gym environment + obs1
register(
    id='widowx_reacher-v1',
    entry_point='widowx_env.envs.1_widowx_pybullet_fixed_gymEnv_obs1:WidowxEnv',
    max_episode_steps=100)

# Pybullet environment + fixed goal + goal environment + obs1
register(
    id='widowx_reacher-v2',
    entry_point='widowx_env.envs.2_widowx_pybullet_fixed_goalEnv_obs1:WidowxEnv',
    max_episode_steps=100)

# Pybullet environment + random goal + gym environment + obs1
register(
    id='widowx_reacher-v3',
    entry_point='widowx_env.envs.3_widowx_pybullet_random_gymEnv_obs1:WidowxEnv',
    max_episode_steps=100)

# Pybullet environment + random goal + goal environment + obs1
register(
    id='widowx_reacher-v4',
    entry_point='widowx_env.envs.4_widowx_pybullet_random_goalEnv_obs1:WidowxEnv',
    max_episode_steps=100)

# Pybullet environment + fixed goal + gym environment + obs2
register(
    id='widowx_reacher-v5',
    entry_point='widowx_env.envs.5_widowx_pybullet_fixed_gymEnv_obs2:WidowxEnv',
    max_episode_steps=100)

# Pybullet environment + fixed goal + gym environment + obs3
register(
    id='widowx_reacher-v6',
    entry_point='widowx_env.envs.6_widowx_pybullet_fixed_gymEnv_obs3:WidowxEnv',
    max_episode_steps=100)

# Pybullet environment + fixed goal + gym environment + obs4
register(
    id='widowx_reacher-v7',
    entry_point='widowx_env.envs.7_widowx_pybullet_fixed_gymEnv_obs4:WidowxEnv',
    max_episode_steps=100)

# Pybullet environment + fixed goal + gym environment + obs5
register(
    id='widowx_reacher-v8',
    entry_point='widowx_env.envs.8_widowx_pybullet_fixed_gymEnv_obs5:WidowxEnv',
    max_episode_steps=100)

# Pybullet environment + random goal + gym environment + obs2
register(
    id='widowx_reacher-v9',
    entry_point='widowx_env.envs.9_widowx_pybullet_random_gymEnv_obs2:WidowxEnv',
    max_episode_steps=100)

# Pybullet environment + random goal + gym environment + obs3
register(
    id='widowx_reacher-v10',
    entry_point='widowx_env.envs.10_widowx_pybullet_random_gymEnv_obs3:WidowxEnv',
    max_episode_steps=100)


# Pybullet environment + random goal + gym environment + obs4
register(
    id='widowx_reacher-v11',
    entry_point='widowx_env.envs.11_widowx_pybullet_random_gymEnv_obs4:WidowxEnv',
    max_episode_steps=100)


# Pybullet environment + random goal + gym environment + obs5
register(
    id='widowx_reacher-v12',
    entry_point='widowx_env.envs.12_widowx_pybullet_random_gymEnv_obs5:WidowxEnv',
    max_episode_steps=100)







# # Pybullet environment + fixed goal + gym environment + obs2
# register(id='widowx_reacher-v2',
#          entry_point='widowx_env.envs.1_widowx_pybullet_fixed_gymEnv_obs2:WidowxEnv',
#          max_episode_steps=100
#          )

# # Pybullet environment + fixed goal + gym environment + obs3
# register(id='widowx_reacher-v3',
#          entry_point='widowx_env.envs.1_widowx_pybullet_fixed_gymEnv_obs3:WidowxEnv',
#          max_episode_steps=100
#          )

# # Pybullet environment + fixed goal + gym environment + obs4
# register(id='widowx_reacher-v4',
#          entry_point='widowx_env.envs.1_widowx_pybullet_fixed_gymEnv_obs4:WidowxEnv',
#          max_episode_steps=100
#          )

# # Pybullet environment + fixed goal + gym environment + obs5
# register(id='widowx_reacher-v5',
#          entry_point='widowx_env.envs.1_widowx_pybullet_fixed_gymEnv_obs5:WidowxEnv',
#          max_episode_steps=100
#          )

# #############

# # Pybullet environment + fixed goal + gym environment + reward 2
# register(id='widowx_reacher-v9',
#          entry_point='widowx_env.envs.5_widowx_pybullet_fixed_gymEnv_reward2:WidowxEnv',
#          max_episode_steps=100
#          )
