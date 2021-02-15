""" Log env kwargs in envs_list.csv file """

import pandas as pd



d = {
    'widowx_reacher-v1':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 1,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': '[-1, -1, -1, -1, -1, -1]',
        'action_max': '[1, 1, 1, 1, 1, 1]',
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v2':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 2,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v3':
        {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 3,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v4':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 4,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v5':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v6':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-0.05, -0.025, -0.025, -0.025, -0.05, -0.0005],
        'action_max': [0.05, 0.025, 0.025, 0.025, 0.05, 0.0005],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v7':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-2, -2, -2, -2, -2, -2],
        'action_max': [2, 2, 2, 2, 2, 2],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v8':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-10, -10, -10, -10, -10, -10],
        'action_max': [10, 10, 10, 10, 10, 10],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v9':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 1000,
    },

    'widowx_reacher-v10':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1, -1, -1, -1, -1, -1],
        'action_max': [1, 1, 1, 1, 1, 1],
        'alpha_reward': 0.1,
        'reward_coeff': 0.001,
        },

    'widowx_reacher-v11':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v12':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 1,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-3, -3, -3, -3, -3, -3],
        'action_max': [3, 3, 3, 3, 3, 3],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v13':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 2,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v14':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 3,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v15':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 4,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v16':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 5,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v17':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 6,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v18':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 7,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v19':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 8,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v20':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 9,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v21':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 10,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v22':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 11,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v23':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 12,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v24':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 13,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v25':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 14,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v26':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 15,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },

    'widowx_reacher-v27':
    {
        'random_goal' : False,
        'goal_oriented' : False,
        'obs_type' : 5,
        'reward_type' : 16,
        'action_type' : 1,
        'joint_limits' : "large",
        'action_min': [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5],
        'action_max': [1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
        'alpha_reward': 0.1,
        'reward_coeff': 1,
        },
}

df = pd.DataFrame.from_dict(d, orient='index').reset_index()
df.rename(columns={'index': 'env_id'}, inplace=True)

print(df)

df.to_csv('gym_envs/widowx_env/envs_list.csv', index=False)
