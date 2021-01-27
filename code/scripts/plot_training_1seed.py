"""
Plot the learning curves for each seed in the log folder
"""

import argparse
import pandas as pd
from matplotlib import pyplot as plt
from stable_baselines3.common.results_plotter import X_EPISODES, X_TIMESTEPS, X_WALLTIME
from stable_baselines3.common.results_plotter import plot_results, load_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--log-folder',
        help='Log folder',
        type=str,
        default='logs')
    parser.add_argument(
        "--env",
        help="environment ID",
        type=str,
        default="CartPole-v1")
    args = parser.parse_args()

    log_dir = args.log_folder

    # Load results
    W = load_results(log_dir)
    # print("Results seed: ", W)

    # Save walltime to stats.csv
    df = pd.read_csv(log_dir + 'stats.csv')
    df["Train walltime (s)"] = W["t"].max()
    df.to_csv(log_dir + "stats.csv", index=False)
    # print(df)

    # Plot training rewards

    TIMESTEPS = 1e10

    plot_results([log_dir], TIMESTEPS, X_TIMESTEPS, args.env)
    plt.savefig(log_dir + "reward_vs_timesteps.png")
    # plt.show()

    plot_results([log_dir], TIMESTEPS, X_EPISODES, args.env)
    plt.savefig(log_dir + "reward_vs_episodes.png")
    # plt.show()

    plot_results([log_dir], TIMESTEPS, X_WALLTIME, args.env)
    plt.savefig(log_dir + "reward_vs_walltime.png")
    # plt.show()
