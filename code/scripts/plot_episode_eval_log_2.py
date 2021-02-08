""" plot episode evaluation log and save """

from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-id', help='Experiment ID', type=int)
    args = parser.parse_args()

    LOG_DIR = "logs/exp_"+str(args.exp_id)

    FILE_PATH = str(list(Path(LOG_DIR).rglob('res_episode_1.csv'))[0])
    SAVE_PATH = FILE_PATH.replace("res_episode_1.csv", "plot_episode_eval_log_2.png")

    # # read data
    log_df = pd.read_csv(FILE_PATH)

    # plot
    fig, axs = plt.subplots(2, 3, figsize=(10, 5), dpi=300, sharex=True)

    log_df.plot(x='timestep', y='normalized_action_1', ax=axs[0, 0])
    log_df.plot(x='timestep', y='normalized_action_2', ax=axs[0, 1])
    log_df.plot(x='timestep', y='normalized_action_3', ax=axs[0, 2])
    log_df.plot(x='timestep', y='normalized_action_4', ax=axs[1, 0])
    log_df.plot(x='timestep', y='normalized_action_5', ax=axs[1, 1])
    log_df.plot(x='timestep', y='normalized_action_6', ax=axs[1, 2])

    axs[0, 0].set_ylabel("normalized action 1")
    axs[0, 1].set_ylabel("normalized action 2")
    axs[0, 2].set_ylabel("normalized action 3")
    axs[1, 0].set_ylabel("normalized action 4")
    axs[1, 1].set_ylabel("normalized action 5")
    axs[1, 2].set_ylabel("normalized action 6")


    plt.tight_layout()
    # plt.show()
    plt.savefig(SAVE_PATH, bbox_inches='tight')
