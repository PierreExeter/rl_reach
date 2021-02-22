"""
Log all metrics and environment variables in benchmark/benchmark_results.csv
Plot learning curves of all the seed runs in the experiment
"""

import sys
import argparse
import os
from pathlib import Path
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results


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
        print(("The environment specified is missing! "
            "Please update gym_envs/widowx_env/envs_list.csv. Exiting..."))
        sys.exit(0)

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
    COUNT = 0

    for filename in res_file_list:
        df = pd.read_csv(filename, index_col=None, header=0)
        df['seed'] = COUNT
        df['log_dir'] = filename
        li.append(df)
        COUNT += 1

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
        'mean_SR_pos_50': df['SR pos 50mm'].mean(),
        'std_SR_pos_50': df['SR pos 50mm'].std(),
        'max_SR_pos_50': df['SR pos 50mm'].max(),
        'mean_RT_pos_50': df['RT pos 50mm'].mean(),
        'std_RT_pos_50': df['RT pos 50mm'].std(),
        'max_RT_pos_50': df['RT pos 50mm'].max(),
        'mean_SR_pos_20': df['SR pos 20mm'].mean(),
        'std_SR_pos_20': df['SR pos 20mm'].std(),
        'max_SR_pos_20': df['SR pos 20mm'].max(),
        'mean_RT_pos_20': df['RT pos 20mm'].mean(),
        'std_RT_pos_20': df['RT pos 20mm'].std(),
        'max_RT_pos_20': df['RT pos 20mm'].max(),
        'mean_SR_pos_10': df['SR pos 10mm'].mean(),
        'std_SR_pos_10': df['SR pos 10mm'].std(),
        'max_SR_pos_10': df['SR pos 10mm'].max(),
        'mean_RT_pos_10': df['RT pos 10mm'].mean(),
        'std_RT_pos_10': df['RT pos 10mm'].std(),
        'max_RT_pos_10': df['RT pos 10mm'].max(),
        'mean_SR_pos_5': df['SR pos 5mm'].mean(),
        'std_SR_pos_5': df['SR pos 5mm'].std(),
        'max_SR_pos_5': df['SR pos 5mm'].max(),
        'mean_RT_pos_5': df['RT pos 5mm'].mean(),
        'std_RT_pos_5': df['RT pos 5mm'].std(),
        'max_RT_pos_5': df['RT pos 5mm'].max(),
        'mean_SR_pos_2': df['SR pos 2mm'].mean(),
        'std_SR_pos_2': df['SR pos 2mm'].std(),
        'max_SR_pos_2': df['SR pos 2mm'].max(),
        'mean_RT_pos_2': df['RT pos 2mm'].mean(),
        'std_RT_pos_2': df['RT pos 2mm'].std(),
        'max_RT_pos_2': df['RT pos 2mm'].max(),
        'mean_SR_pos_1': df['SR pos 1mm'].mean(),
        'std_SR_pos_1': df['SR pos 1mm'].std(),
        'max_SR_pos_1': df['SR pos 1mm'].max(),
        'mean_RT_pos_1': df['RT pos 1mm'].mean(),
        'std_RT_pos_1': df['RT pos 1mm'].std(),
        'max_RT_pos_1': df['RT pos 1mm'].max(),
        'mean_SR_pos_05': df['SR pos 0.5mm'].mean(),
        'std_SR_pos_05': df['SR pos 0.5mm'].std(),
        'max_SR_pos_05': df['SR pos 0.5mm'].max(),
        'mean_RT_pos_05': df['RT pos 0.5mm'].mean(),
        'std_RT_pos_05': df['RT pos 0.5mm'].std(),
        'max_RT_pos_05': df['RT pos 0.5mm'].max(),
        'mean_SR_orient_50': df['SR orient 50mm'].mean(),
        'std_SR_orient_50': df['SR orient 50mm'].std(),
        'max_SR_orient_50': df['SR orient 50mm'].max(),
        'mean_RT_orient_50': df['RT orient 50mm'].mean(),
        'std_RT_orient_50': df['RT orient 50mm'].std(),
        'max_RT_orient_50': df['RT orient 50mm'].max(),
        'mean_SR_orient_20': df['SR orient 20mm'].mean(),
        'std_SR_orient_20': df['SR orient 20mm'].std(),
        'max_SR_orient_20': df['SR orient 20mm'].max(),
        'mean_RT_orient_20': df['RT orient 20mm'].mean(),
        'std_RT_orient_20': df['RT orient 20mm'].std(),
        'max_RT_orient_20': df['RT orient 20mm'].max(),
        'mean_SR_orient_10': df['SR orient 10mm'].mean(),
        'std_SR_orient_10': df['SR orient 10mm'].std(),
        'max_SR_orient_10': df['SR orient 10mm'].max(),
        'mean_RT_orient_10': df['RT orient 10mm'].mean(),
        'std_RT_orient_10': df['RT orient 10mm'].std(),
        'max_RT_orient_10': df['RT orient 10mm'].max(),
        'mean_SR_orient_5': df['SR orient 5mm'].mean(),
        'std_SR_orient_5': df['SR orient 5mm'].std(),
        'max_SR_orient_5': df['SR orient 5mm'].max(),
        'mean_RT_orient_5': df['RT orient 5mm'].mean(),
        'std_RT_orient_5': df['RT orient 5mm'].std(),
        'max_RT_orient_5': df['RT orient 5mm'].max(),
        'mean_SR_orient_2': df['SR orient 2mm'].mean(),
        'std_SR_orient_2': df['SR orient 2mm'].std(),
        'max_SR_orient_2': df['SR orient 2mm'].max(),
        'mean_RT_orient_2': df['RT orient 2mm'].mean(),
        'std_RT_orient_2': df['RT orient 2mm'].std(),
        'max_RT_orient_2': df['RT orient 2mm'].max(),
        'mean_SR_orient_1': df['SR orient 1mm'].mean(),
        'std_SR_orient_1': df['SR orient 1mm'].std(),
        'max_SR_orient_1': df['SR orient 1mm'].max(),
        'mean_RT_orient_1': df['RT orient 1mm'].mean(),
        'std_RT_orient_1': df['RT orient 1mm'].std(),
        'max_RT_orient_1': df['RT orient 1mm'].max(),
        'mean_SR_orient_05': df['SR orient 0.5mm'].mean(),
        'std_SR_orient_05': df['SR orient 0.5mm'].std(),
        'max_SR_orient_05': df['SR orient 0.5mm'].max(),
        'mean_RT_orient_05': df['RT orient 0.5mm'].mean(),
        'std_RT_orient_05': df['RT orient 0.5mm'].std(),
        'max_RT_orient_05': df['RT orient 0.5mm'].max()
    }

    df_metrics = pd.DataFrame(metrics_dict, index=[0])
    df_metrics.to_csv(log_dir + "results_exp.csv", index=False)

    # concatenate all 3 dictionaries
    benchmark_dict = {**env_dict, **hyperparams_dict, **metrics_dict}

    # transform into a dataframe
    df_bench = pd.DataFrame(benchmark_dict, index=[0])

    BENCH_PATH = "benchmark/benchmark_results.csv"
    if os.path.isfile(BENCH_PATH):

        backedup_df = pd.read_csv(BENCH_PATH)
        # If experiment hasn't been evaluated yet
        if args.exp_id in backedup_df['exp_id'].values:
            ANSWER = None
            while ANSWER not in ("Y", "n"):
                ANSWER = input(("This experiment has already been evaluated and "
                "added to the benchmark file. Do you still want to continue ? [Y/n] "))
                if ANSWER == "Y":
                    break
                if ANSWER == "n":
                    print("Aborting...")
                    sys.exit()
                else:
                    print("Please enter Y or n.")

        # add to existing results and save
        appended_df = backedup_df.append(df_bench, ignore_index=True)
        appended_df.to_csv(BENCH_PATH, index=False)
    else:
        # if benchmark file doesn't exist, save df_bench
        df_bench.to_csv(BENCH_PATH, index=False)

    ###############
    # 2. PLOT LEARNING CURVES
    ###############

    ##### 2.1. create all_reward dataframe

    res_file_list = []

    for path in Path(log_dir).rglob(args.env + '_*'):
        res_file_list.append(path)

    res_file_list = sorted(res_file_list)
    # print(res_file_list)

    df_list = []
    col_list = []
    COUNT = 1

    for filename in res_file_list:
        # print(filename)
        FILENAME = str(filename)  # convert from Posixpath to string

        W = load_results(FILENAME)
        # print(W['r'])

        df_list.append(W['r'])
        col_list.append("seed " + str(COUNT))
        COUNT += 1

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

    # apply rolling window (except on timesteps)
    all_rewards_smoothed = all_rewards.copy()

    for col in all_rewards_smoothed.columns[:-1]:
        # print(col)
        all_rewards_smoothed[col] = all_rewards_smoothed[col].rolling(window=50).mean()

    all_rewards_smoothed.dropna(inplace=True)  # remove NaN due to rolling average
    all_rewards_smoothed.to_csv(log_dir + "all_rewards_smooth.csv", index=False)
    # print(all_rewards_smoothed)

    # remove underscore in column name (issue with tex)
    all_rewards.rename(lambda s: s.replace('_', ' '), axis='columns', inplace=True)
    all_rewards_smoothed.rename(lambda s: s.replace('_', ' '), axis='columns', inplace=True)

    ###### 2.2. Plot

    fs = 15
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)

    ##### 2.2.1. plot reward vs timesteps
    fig, ax = plt.subplots(figsize=(10, 7))

    for seed_col in col_list:
        # print(seed_col)
        all_rewards.plot(x='timesteps', y=seed_col, ax=ax)

    all_rewards.plot(x='timesteps', y='mean reward', ax=ax, color='k')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 5))

    plt.xlabel(r'Time steps', fontsize=fs)
    plt.ylabel(r'Average return', fontsize=fs)
    plt.legend(loc="lower right", fontsize=fs)

    plt.savefig(log_dir + "reward_vs_timesteps.png", bbox_inches='tight', dpi=100)
    # plt.show()

    ########## 2.2.2. plot reward vs timesteps (smoothed)
    fig, ax = plt.subplots(figsize=(10, 7))

    for seed_col in col_list:
        # print(seed_col)
        all_rewards_smoothed.plot(x='timesteps', y=seed_col, ax=ax)

    all_rewards_smoothed.plot(x='timesteps', y='mean reward', ax=ax, color='k')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 5))

    plt.xlabel(r'Time steps', fontsize=fs)
    plt.ylabel(r'Average return', fontsize=fs)
    plt.legend(loc="lower right", fontsize=fs)

    plt.savefig(log_dir + "reward_vs_timesteps_smoothed.png", bbox_inches='tight', dpi=100)
    # plt.show()

    # ####### 2.2.3. plot reward vs timesteps (fill between)
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(all_rewards['timesteps'], all_rewards['mean reward'], color='k', label='Mean return', linewidth=0.5)
    ax.fill_between(all_rewards['timesteps'], all_rewards['lower'], all_rewards['upper'], color='r', alpha=0.35, label='Confidence interval')
    # plt.axhline(y=0, color='r', linestyle='--', label="min reward")
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 5))

    plt.xlabel(r'Time steps', fontsize=fs)
    plt.ylabel(r'Average return', fontsize=fs)
    plt.legend(loc="lower right", fontsize=fs)

    plt.savefig(log_dir + "reward_vs_timesteps_fill.png", bbox_inches='tight', dpi=100)
    # plt.show()

    # ####### 2.2.4. plot reward vs timesteps (fill between) + smoothed
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.plot(all_rewards_smoothed['timesteps'], all_rewards_smoothed['mean reward'], color='k', label='Mean return')
    ax.fill_between(all_rewards_smoothed['timesteps'], all_rewards_smoothed['lower'], all_rewards_smoothed['upper'], color='r', alpha=0.35, label='Confidence interval')
    # plt.axhline(y=0, color='r', linestyle='--', label="min reward")
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 5))

    plt.xlabel(r'Time steps', fontsize=fs)
    plt.ylabel(r'Average return', fontsize=fs)
    plt.legend(loc="lower right", fontsize=fs)

    plt.savefig(log_dir + "reward_vs_timesteps_fill_smoothed.png", bbox_inches='tight', dpi=100)
    # plt.show()

