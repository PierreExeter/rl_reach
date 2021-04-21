""" Log env kwargs in envs_list.csv file """

import pandas as pd
from env_kwargs import kwargs_dicts


df = pd.DataFrame.from_dict(kwargs_dicts, orient='index').reset_index()
df.rename(columns={'index': 'env_id'}, inplace=True)

print(df)

df.to_csv('gym_envs/gym_envs/envs_list.csv', index=False)
