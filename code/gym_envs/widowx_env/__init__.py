""" Register Gym environments """

from gym.envs.registration import register


register(
    id='widowx_reacher-v1',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 1,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_coeff' : 30,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v2',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_coeff' : 30,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v3',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : True,
        'goal_oriented' : False,
        'obs_type' : 1,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_coeff' : 30,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v4',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : True,
        'goal_oriented' : True,
        'obs_type' : 1,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_coeff' : 30,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v5',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 2,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_coeff' : 30,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v6',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 3,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_coeff' : 30,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v7',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 4,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_coeff' : 30,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v8',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_coeff' : 30,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v9',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : True,
        'goal_oriented' : False,
        'obs_type' : 2,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_coeff' : 30,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v10',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : True,
        'goal_oriented' : False,
        'obs_type' : 3,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_coeff' : 30,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v11',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : True,
        'goal_oriented' : False,
        'obs_type' : 4,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_coeff' : 30,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v12',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : True,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "small",
        'action_coeff' : 30,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v13',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : True,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': False,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v14',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v15',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 2,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v16',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 3,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v17',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 4,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v18',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 5,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v19',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 6,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v20',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 7,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v21',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 8,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v22',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 9,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 1,
        },
    )

register(
    id='widowx_reacher-v23',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 10,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v24',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 11,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v25',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 12,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.1,
        },
    )

register(
    id='widowx_reacher-v26',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 4,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.01,
        },
    )

register(
    id='widowx_reacher-v27',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 5,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.01,
        },
    )

register(
    id='widowx_reacher-v28',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 4,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.001,
        },
    )

register(
    id='widowx_reacher-v29',
    entry_point='widowx_env.envs.widowx_env:WidowxEnv',
    max_episode_steps=100,
    kwargs={
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 4,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_coeff' : 10,
        'normalize_action': True,
        'alpha': 0.0001,
        },
    )
