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
        'alpha': 0.1,
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
        'alpha': 0.1,
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
        'alpha': 0.1,
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
        'alpha': 0.1,
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
        'alpha': 0.1,
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
        'alpha': 0.1,
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
        'alpha': 0.1,
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
        'alpha': 0.1,
        'reward_coeff': 1,
        },
}


df = pd.DataFrame.from_dict(d, orient='index').reset_index()
df.rename(columns={'index': 'env_id'}, inplace=True)

print(df)

df.to_csv('gym_envs/widowx_env/envs_list.csv', index=False)
