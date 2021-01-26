"""
Plot data from benchmark file
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-list',
        help="List of experiment ID to plot (eg. 1 2 3)",
        nargs="+")
    parser.add_argument(
        '--col',
        help="Column name (hyperparameter name) of the benchmark file",
        type=str)
    args = parser.parse_args()

    # Import data
    df = pd.read_csv("benchmark/benchmark_results.csv")
    # print(df)

    # select experiments ID to plot
    exp_list = args.exp_list
    EXP_LIST_STRING = '_'.join(exp_list) # concatenate list: used for plot title
    exp_list = list(map(int, exp_list))  # convert list of string to list of ints
    df_plot = df[df['exp_id'].isin(exp_list)]

    col = args.col
    # ent_coef is a string and must be converted into a float
    if col == "ent_coef":
        df_plot['ent_coef'] = df_plot['ent_coef'].astype(float)

    # sort col in ascending order
    df_plot = df_plot.sort_values(col)
    print(df_plot)

    # Plot
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 10), dpi=300)

    df_plot.plot(x=col, y='mean_return', yerr='std_return', capsize=4, ax=ax1)
    df_plot.plot(x=col, y='mean_train_time(s)', yerr='std_train_time(s)', capsize=4, ax=ax2)
    df_plot.plot(x=col, y='mean_SR_50', yerr='std_SR_50', capsize=4, ax=ax3)
    df_plot.plot(x=col, y='mean_SR_20', yerr='std_SR_20', capsize=4, ax=ax3)
    df_plot.plot(x=col, y='mean_SR_10', yerr='std_SR_10', capsize=4, ax=ax3)
    df_plot.plot(x=col, y='mean_SR_5', yerr='std_SR_5', capsize=4, ax=ax3)
    df_plot.plot(x=col, y='mean_SR_2', yerr='std_SR_2', capsize=4, ax=ax3)
    df_plot.plot(x=col, y='mean_SR_1', yerr='std_SR_1', capsize=4, ax=ax3)
    df_plot.plot(x=col, y='mean_SR_05', yerr='std_SR_05', capsize=4, ax=ax3)
    df_plot.plot(x=col, y='max_SR_50', marker='x', ax=ax4)
    df_plot.plot(x=col, y='max_SR_20', marker='x', ax=ax4)
    df_plot.plot(x=col, y='max_SR_10', marker='x', ax=ax4)
    df_plot.plot(x=col, y='max_SR_5', marker='x', ax=ax4)
    df_plot.plot(x=col, y='max_SR_2', marker='x', ax=ax4)
    df_plot.plot(x=col, y='max_SR_1', marker='x', ax=ax4)
    df_plot.plot(x=col, y='max_SR_05', marker='x', ax=ax4)
    df_plot.plot(x=col, y='mean_RT_50', yerr='std_RT_50', capsize=4, ax=ax5)
    df_plot.plot(x=col, y='mean_RT_20', yerr='std_RT_20', capsize=4, ax=ax5)
    df_plot.plot(x=col, y='mean_RT_10', yerr='std_RT_10', capsize=4, ax=ax5)
    df_plot.plot(x=col, y='mean_RT_5', yerr='std_RT_5', capsize=4, ax=ax5)
    df_plot.plot(x=col, y='mean_RT_2', yerr='std_RT_2', capsize=4, ax=ax5)
    df_plot.plot(x=col, y='mean_RT_1', yerr='std_RT_1', capsize=4, ax=ax5)
    df_plot.plot(x=col, y='mean_RT_05', yerr='std_RT_05', capsize=4, ax=ax5)

    ax1.set_ylabel("Mean return")
    ax2.set_ylabel("Train time (s)")
    ax3.set_ylabel("Mean success ratio")
    ax4.set_ylabel("Max success ratio")
    ax5.set_ylabel("Reach time")

    # uncomment if var = n_timesteps (clearer plot)
    ## ax3.set_xscale('symlog', linthreshy=1e-1)
    ## ax3.set_yscale('symlog', linthreshy=1e-1)
    # ax3.set_xscale('log')
    # ax3.set_yscale('log')

    ax3.legend(bbox_to_anchor=(1, 1.05))
    ax4.legend(bbox_to_anchor=(1, 1.05))
    ax5.legend(bbox_to_anchor=(1.2, 1.05))

    plt.tight_layout()
    # plt.show()
    plt.savefig("benchmark/plots/" + col + "_exp_" + EXP_LIST_STRING + ".png", bbox_inches='tight')
