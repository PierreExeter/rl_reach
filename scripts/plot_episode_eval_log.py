""" plot episode evaluation log and save """

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-id', help='Experiment ID', type=int)
    args = parser.parse_args()

    log_dir = "logs/exp_"+str(args.exp_id)

    file_path = str(list(Path(log_dir).rglob('res_episode_1.csv'))[0])
    save_path = file_path.replace("res_episode_1.csv", "plot_episode_eval_log.png")

    # # read data
    log_df = pd.read_csv(file_path)

    # plot
    fig, axs = plt.subplots(4, 4, figsize=(20, 10), dpi=300, sharex=True)

    log_df.plot(x='timestep', y='joint_pos1', ax=axs[0, 0])
    log_df.plot(x='timestep', y='joint1_min', ax=axs[0, 0], style="r--")
    log_df.plot(x='timestep', y='joint1_max', ax=axs[0, 0], style="r--")

    log_df.plot(x='timestep', y='joint_pos2', ax=axs[1, 0])
    log_df.plot(x='timestep', y='joint2_min', ax=axs[1, 0], style="r--")
    log_df.plot(x='timestep', y='joint2_max', ax=axs[1, 0], style="r--")

    log_df.plot(x='timestep', y='joint_pos3', ax=axs[2, 0])
    log_df.plot(x='timestep', y='joint3_min', ax=axs[2, 0], style="r--")
    log_df.plot(x='timestep', y='joint3_max', ax=axs[2, 0], style="r--")

    log_df.plot(x='timestep', y='joint_pos4', ax=axs[3, 0])
    log_df.plot(x='timestep', y='joint4_min', ax=axs[3, 0], style="r--")
    log_df.plot(x='timestep', y='joint4_max', ax=axs[3, 0], style="r--")

    log_df.plot(x='timestep', y='joint_pos5', ax=axs[0, 2])
    log_df.plot(x='timestep', y='joint5_min', ax=axs[0, 2], style="r--")
    log_df.plot(x='timestep', y='joint5_max', ax=axs[0, 2], style="r--")

    log_df.plot(x='timestep', y='joint_pos6', ax=axs[1, 2])
    log_df.plot(x='timestep', y='joint6_min', ax=axs[1, 2], style="r--")
    log_df.plot(x='timestep', y='joint6_max', ax=axs[1, 2], style="r--")

    log_df.plot(x='timestep', y='action_1', ax=axs[0, 1])
    log_df.plot(x='timestep', y='action_low1', ax=axs[0, 1], style="r--")
    log_df.plot(x='timestep', y='action_high1', ax=axs[0, 1], style="r--")

    log_df.plot(x='timestep', y='action_2', ax=axs[1, 1])
    log_df.plot(x='timestep', y='action_low2', ax=axs[1, 1], style="r--")
    log_df.plot(x='timestep', y='action_high2', ax=axs[1, 1], style="r--")

    log_df.plot(x='timestep', y='action_3', ax=axs[2, 1])
    log_df.plot(x='timestep', y='action_low3', ax=axs[2, 1], style="r--")
    log_df.plot(x='timestep', y='action_high3', ax=axs[2, 1], style="r--")

    log_df.plot(x='timestep', y='action_4', ax=axs[3, 1])
    log_df.plot(x='timestep', y='action_low4', ax=axs[3, 1], style="r--")
    log_df.plot(x='timestep', y='action_high4', ax=axs[3, 1], style="r--")

    log_df.plot(x='timestep', y='action_5', ax=axs[0, 3])
    log_df.plot(x='timestep', y='action_low5', ax=axs[0, 3], style="r--")
    log_df.plot(x='timestep', y='action_high5', ax=axs[0, 3], style="r--")

    log_df.plot(x='timestep', y='action_6', ax=axs[1, 3])
    log_df.plot(x='timestep', y='action_low6', ax=axs[1, 3], style="r--")
    log_df.plot(x='timestep', y='action_high6', ax=axs[1, 3], style="r--")

    log_df.plot(x='timestep', y='reward', ax=axs[2, 2], color="b")
    ax_1 = axs[2, 2].twinx()
    log_df.plot(x='timestep', y='return', ax=ax_1, color="r")

    log_df.plot(x='timestep', y='distance',
                ax=axs[2, 3], color="b", marker="x")

    log_df.plot(x='timestep', y='est_acc', ax=axs[3, 2], color="g", marker="*")
    ax_3 = axs[3, 2].twinx()
    log_df.plot(x='timestep', y='est_vel', ax=ax_3, color="r", marker="+")

    log_df.plot(x='timestep', y='goal_x', ax=axs[3, 3], style='or')
    log_df.plot(x='timestep', y='goal_y', ax=axs[3, 3], style='ob')
    log_df.plot(x='timestep', y='goal_z', ax=axs[3, 3], style='og')
    log_df.plot(x='timestep', y='tip_x', ax=axs[3, 3], style='xr')
    log_df.plot(x='timestep', y='tip_y', ax=axs[3, 3], style='xb')
    log_df.plot(x='timestep', y='tip_z', ax=axs[3, 3], style='xg')

    axs[0, 0].set_ylabel("joint1 pos (rad)")
    axs[1, 0].set_ylabel("joint2 pos (rad)")
    axs[2, 0].set_ylabel("joint3 pos (rad)")
    axs[3, 0].set_ylabel("joint4 pos (rad)")
    axs[0, 2].set_ylabel("joint5 pos (rad)")
    axs[1, 2].set_ylabel("joint6 pos (rad)")

    axs[0, 1].set_ylabel("action1 (rad)")
    axs[1, 1].set_ylabel("action1 (rad)")
    axs[2, 1].set_ylabel("action1 (rad)")
    axs[3, 1].set_ylabel("action1 (rad)")
    axs[0, 3].set_ylabel("action1 (rad)")
    axs[1, 3].set_ylabel("action1 (rad)")

    axs[2, 2].set_ylabel("reward (m^2)", color="b")
    ax_1.set_ylabel("return (m^2)", color="r")
    axs[2, 2].tick_params(axis='y', labelcolor="b")
    ax_1.tick_params(axis='y', labelcolor="r")

    axs[2, 3].set_ylabel("distance (m)")
    # axs[2, 3].tick_params(axis='y', labelcolor="b")

    axs[3, 2].set_ylabel("acc (m/s^2)", color="g")
    ax_3.set_ylabel("vel (m/s)", color="r")
    axs[3, 2].tick_params(axis='y', labelcolor="g")
    ax_3.tick_params(axis='y', labelcolor="r")

    axs[3, 3].set_ylabel("coordinates (m)")

    axs[3, 3].legend(loc="upper right")
    # ax3.legend(bbox_to_anchor=(1, 1.05))
    # ax4.legend(bbox_to_anchor=(1.2, 1.05))

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)
