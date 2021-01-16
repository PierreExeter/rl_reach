"""
Log all metrics and environment variables in benchmark/benchmark_results.csv
Plot learning curves of all the seed runs in the experiment
"""

import argparse
import os
import pandas as pd
import yaml
import sys
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3.common.results_plotter import load_results
from sys import exit


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str)
    parser.add_argument('-e', '--env', help='env name', type=str)
    parser.add_argument('-a', '--algo', help='RL Algorithm', type=str)
    parser.add_argument('-ns', '--nb-seeds', help='number of seeds', type=int)
    parser.add_argument(
        '-n',
        '--n-eval-steps',
        help="Number of evaluation timesteps",
        type=int)
    parser.add_argument(
        '-d',
        '--deterministic-flag',
        help='0: Stochastic evaluation, 1: deterministic evaluation',
        type=int)
    parser.add_argument('--exp-id', help='Experiment ID', type=int)
    args = parser.parse_args()

    log_dir = args.log_folder + "/" + args.algo + "/"
    # print(log_dir)

    ###############
    # 1. SAVE ENVS VARIABLES, TRAINING HYPERPARAMETERS AND EVALUATION METRICS TO BENCHMARK FILE
    ###############

    # 1.1. Create dict of useful environment variables

    try:
        df_envs = pd.read_csv("gym_envs/widowx_env/envs_list.csv")
        # print(df_envs)
        df_env = df_envs[df_envs["env_id"] == args.env]
        env_dict = df_env.to_dict('records')[0]
        # print(env_dict)
    except BaseException:
        print("The environment specified is missing! Please update gym_envs/widowx_env/envs_list.csv. Exiting...")
        exit(0)

    env_dict['exp_id'] = args.exp_id
    env_dict['algo'] = args.algo
    env_dict['nb_seeds'] = args.nb_seeds
    env_dict['nb_eval_timesteps'] = args.n_eval_steps
    env_dict['deterministic'] = args.deterministic_flag

    # 1.2. Create hyperparams dict

    # Load hyperparameters from config.yml
    config_path = list(Path(log_dir).rglob('config.yml'))[0]

    with open(config_path, 'r') as f:
        hyperparams_ordered = yaml.load(f, Loader=yaml.UnsafeLoader)
        hyperparams_dict = dict(hyperparams_ordered)

    # Load extra training arguments from args.yml
    args_path = list(Path(log_dir).rglob('args.yml'))[0]

    with open(args_path, 'r') as f:
        args_ordered = yaml.load(f, Loader=yaml.UnsafeLoader)
        args_dict = dict(args_ordered)

    # append useful entries to hyperparams_dict
    hyperparams_dict['nb_eval_ep_during_training'] = args_dict['eval_episodes']
    hyperparams_dict['eval_freq_during_training'] = args_dict['eval_freq']
    hyperparams_dict['vec_env'] = args_dict['vec_env']
    # overwrite n_timesteps from hyperparams.yml
    hyperparams_dict['n_timesteps'] = args_dict['n_timesteps']
    hyperparams_dict['num_threads'] = args_dict['num_threads']

    # print(hyperparams_dict)

    # 1.3. create metric dict

    res_file_list = []

    for path in Path(log_dir).rglob('stats.csv'):
        # print(path)
        res_file_list.append(path)

    res_file_list = sorted(res_file_list)
    # print(res_file_list)

    li = []
    count = 0

    for filename in res_file_list:
        df = pd.read_csv(filename, index_col=None, header=0)
        df['seed'] = count
        df['log_dir'] = filename
        li.append(df)
        count += 1

    # print(li)

    df = pd.concat(li, axis=0, ignore_index=True)
    # print(df)

    metrics_dict = {
        'mean_train_time(s)': df['Train walltime (s)'].mean(),
        'std_train_time(s)': df['Train walltime (s)'].std(),
        'min_train_time(s)': df['Train walltime (s)'].min(),
        'simulated_time(s)': hyperparams_dict['n_timesteps'] / 240,
        'mean_return': df['Eval mean reward'].mean(),
        'std_return': df['Eval mean reward'].std(),
        'max_return': df['Eval mean reward'].max(),
        'mean_SR_50': df['success ratio 50mm'].mean(),
        'std_SR_50': df['success ratio 50mm'].std(),
        'max_SR_50': df['success ratio 50mm'].max(),
        'mean_RT_50': df['Average reach time 50mm'].mean(),
        'std_RT_50': df['Average reach time 50mm'].std(),
        'max_RT_50': df['Average reach time 50mm'].max(),
        'mean_SR_20': df['success ratio 20mm'].mean(),
        'std_SR_20': df['success ratio 20mm'].std(),
        'max_SR_20': df['success ratio 20mm'].max(),
        'mean_RT_20': df['Average reach time 20mm'].mean(),
        'std_RT_20': df['Average reach time 20mm'].std(),
        'max_RT_20': df['Average reach time 20mm'].max(),
        'mean_SR_10': df['success ratio 10mm'].mean(),
        'std_SR_10': df['success ratio 10mm'].std(),
        'max_SR_10': df['success ratio 10mm'].max(),
        'mean_RT_10': df['Average reach time 10mm'].mean(),
        'std_RT_10': df['Average reach time 10mm'].std(),
        'max_RT_10': df['Average reach time 10mm'].max(),
        'mean_SR_5': df['success ratio 5mm'].mean(),
        'std_SR_5': df['success ratio 5mm'].std(),
        'max_SR_5': df['success ratio 5mm'].max(),
        'mean_RT_5': df['Average reach time 5mm'].mean(),
        'std_RT_5': df['Average reach time 5mm'].std(),
        'max_RT_5': df['Average reach time 5mm'].max(),
        'mean_SR_2': df['success ratio 2mm'].mean(),
        'std_SR_2': df['success ratio 2mm'].std(),
        'max_SR_2': df['success ratio 2mm'].max(),
        'mean_RT_2': df['Average reach time 2mm'].mean(),
        'std_RT_2': df['Average reach time 2mm'].std(),
        'max_RT_2': df['Average reach time 2mm'].max(),
        'mean_SR_1': df['success ratio 1mm'].mean(),
        'std_SR_1': df['success ratio 1mm'].std(),
        'max_SR_1': df['success ratio 1mm'].max(),
        'mean_RT_1': df['Average reach time 1mm'].mean(),
        'std_RT_1': df['Average reach time 1mm'].std(),
        'max_RT_1': df['Average reach time 1mm'].max(),
        'mean_SR_05': df['success ratio 0.5mm'].mean(),
        'std_SR_05': df['success ratio 0.5mm'].std(),
        'max_SR_05': df['success ratio 0.5mm'].max(),
        'mean_RT_05': df['Average reach time 0.5mm'].mean(),
        'std_RT_05': df['Average reach time 0.5mm'].std(),
        'max_RT_05': df['Average reach time 0.5mm'].max()
    }

    df_metrics = pd.DataFrame(metrics_dict, index=[0])
    df_metrics.to_csv(log_dir + "results_exp.csv", index=False)

    # concatenate all 3 dictionaries
    benchmark_dict = {**env_dict, **hyperparams_dict, **metrics_dict}

    # transform into a dataframe
    df_bench = pd.DataFrame(benchmark_dict, index=[0])

    bench_path = "benchmark/benchmark_results.csv"
    if os.path.isfile(bench_path):

        backedup_df = pd.read_csv(bench_path)
        # If experiment hasn't been evaluated yet
        if args.exp_id in backedup_df['exp_id'].values:
            answer = None
            while answer not in ("Y", "n"):
                answer = input(
                    "This experiment has already been evaluated and added to the benchmark file. Do you still want to continue ? [Y/n] ")
                if answer == "Y":
                    break
                elif answer == "n":
                    print("Aborting...")
                    sys.exit()
                else:
                    print("Please enter Y or n.")

        # add to existing results and save
        appended_df = backedup_df.append(df_bench, ignore_index=True)
        appended_df.to_csv(bench_path, index=False)
    else:
        # if benchmark file doesn't exist, save df_bench
        df_bench.to_csv(bench_path, index=False)

    ###############
    # 2. PLOT LEARNING CURVES OF ALL THE SEED RUNS IN THE EXPERIMENT
    ###############

    # Plot the learning curve of all the seed runs in the experiment

    res_file_list = []

    for path in Path(log_dir).rglob(args.env + '_*'):
        res_file_list.append(path)

    res_file_list = sorted(res_file_list)
    # print(res_file_list)

    df_list = []
    col_list = []
    count = 1

    for filename in res_file_list:
        # print(filename)
        filename = str(filename)  # convert from Posixpath to string

        W = load_results(filename)
        # print(W['r'])

        df_list.append(W['r'])
        col_list.append("seed " + str(count))
        count += 1

    all_rewards = pd.concat(df_list, axis=1)
    all_rewards.columns = col_list

    all_rewards_copy = all_rewards.copy()
    all_rewards["mean_reward"] = all_rewards_copy.mean(axis=1)
    all_rewards["std_reward"] = all_rewards_copy.std(axis=1)
    all_rewards["upper"] = all_rewards["mean_reward"] + \
        all_rewards["std_reward"]
    all_rewards["lower"] = all_rewards["mean_reward"] - \
        all_rewards["std_reward"]
    all_rewards['timesteps'] = W['l'].cumsum()
    all_rewards.to_csv(log_dir + "all_rewards.csv", index=False)

    # plot
    plt.figure(1, figsize=(10, 5))
    ax = plt.axes()

    for seed_col in col_list:
        # print(seed_col)
        all_rewards.plot(x='timesteps', y=seed_col, ax=ax)

    all_rewards.plot(x='timesteps', y='mean_reward', ax=ax, color='k')

    plt.xlabel('Time steps')
    plt.ylabel('Rewards')

    plt.legend()
    plt.savefig(log_dir + "reward_vs_timesteps.png", dpi=100)
    # plt.show()

    # apply rolling window (except on timesteps)
    for col in all_rewards.columns[:-1]:
        # print(col)
        all_rewards[col] = all_rewards[col].rolling(window=50).mean()

    all_rewards.dropna(inplace=True)  # remove NaN due to rolling average
    all_rewards.to_csv(log_dir + "all_rewards_smooth.csv", index=False)
    # print(all_rewards)

    # plot
    plt.figure(2, figsize=(10, 5))
    ax = plt.axes()

    for seed_col in col_list:
        # print(seed_col)
        all_rewards.plot(x='timesteps', y=seed_col, ax=ax)

    all_rewards.plot(x='timesteps', y='mean_reward', ax=ax, color='k')

    plt.xlabel('Time steps')
    plt.ylabel('Rewards')

    plt.legend()
    plt.savefig(log_dir + "reward_vs_timesteps_smoothed.png", dpi=100)
    # plt.show()
