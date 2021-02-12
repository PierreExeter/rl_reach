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
    SAVE_PATH = FILE_PATH.replace("res_episode_1.csv", "plot_episode_eval_log.png")

    # # read data
    log_df = pd.read_csv(FILE_PATH)

    # plot
    fig, axs = plt.subplots(4, 6, figsize=(25, 10), dpi=300, sharex=True)

    log_df.plot(x='timestep', y='joint1_pos', ax=axs[0, 0])
    log_df.plot(x='timestep', y='joint1_min', ax=axs[0, 0], style="r--")
    log_df.plot(x='timestep', y='joint1_max', ax=axs[0, 0], style="r--")

    log_df.plot(x='timestep', y='joint2_pos', ax=axs[0, 1])
    log_df.plot(x='timestep', y='joint2_min', ax=axs[0, 1], style="r--")
    log_df.plot(x='timestep', y='joint2_max', ax=axs[0, 1], style="r--")

    log_df.plot(x='timestep', y='joint3_pos', ax=axs[0, 2])
    log_df.plot(x='timestep', y='joint3_min', ax=axs[0, 2], style="r--")
    log_df.plot(x='timestep', y='joint3_max', ax=axs[0, 2], style="r--")

    log_df.plot(x='timestep', y='joint4_pos', ax=axs[0, 3])
    log_df.plot(x='timestep', y='joint4_min', ax=axs[0, 3], style="r--")
    log_df.plot(x='timestep', y='joint4_max', ax=axs[0, 3], style="r--")

    log_df.plot(x='timestep', y='joint5_pos', ax=axs[0, 4])
    log_df.plot(x='timestep', y='joint5_min', ax=axs[0, 4], style="r--")
    log_df.plot(x='timestep', y='joint5_max', ax=axs[0, 4], style="r--")

    log_df.plot(x='timestep', y='joint6_pos', ax=axs[0, 5])
    log_df.plot(x='timestep', y='joint6_min', ax=axs[0, 5], style="r--")
    log_df.plot(x='timestep', y='joint6_max', ax=axs[0, 5], style="r--")

    log_df.plot(x='timestep', y='action1', ax=axs[1, 0])
    log_df.plot(x='timestep', y='action1_min', ax=axs[1, 0], style="r--")
    log_df.plot(x='timestep', y='action1_max', ax=axs[1, 0], style="r--")

    log_df.plot(x='timestep', y='action2', ax=axs[1, 1])
    log_df.plot(x='timestep', y='action2_min', ax=axs[1, 1], style="r--")
    log_df.plot(x='timestep', y='action2_max', ax=axs[1, 1], style="r--")

    log_df.plot(x='timestep', y='action3', ax=axs[1, 2])
    log_df.plot(x='timestep', y='action3_min', ax=axs[1, 2], style="r--")
    log_df.plot(x='timestep', y='action3_max', ax=axs[1, 2], style="r--")

    log_df.plot(x='timestep', y='action4', ax=axs[1, 3])
    log_df.plot(x='timestep', y='action4_min', ax=axs[1, 3], style="r--")
    log_df.plot(x='timestep', y='action4_max', ax=axs[1, 3], style="r--")

    log_df.plot(x='timestep', y='action5', ax=axs[1, 4])
    log_df.plot(x='timestep', y='action5_min', ax=axs[1, 4], style="r--")
    log_df.plot(x='timestep', y='action5_max', ax=axs[1, 4], style="r--")

    log_df.plot(x='timestep', y='action6', ax=axs[1, 5])
    log_df.plot(x='timestep', y='action6_min', ax=axs[1, 5], style="r--")
    log_df.plot(x='timestep', y='action6_max', ax=axs[1, 5], style="r--")

    log_df.plot(x='timestep', y='pyb_action1', ax=axs[2, 0])
    log_df.plot(x='timestep', y='pyb_action1_min', ax=axs[2, 0], style="r--")
    log_df.plot(x='timestep', y='pyb_action1_max', ax=axs[2, 0], style="r--")

    log_df.plot(x='timestep', y='pyb_action2', ax=axs[2, 1])
    log_df.plot(x='timestep', y='pyb_action2_min', ax=axs[2, 1], style="r--")
    log_df.plot(x='timestep', y='pyb_action2_max', ax=axs[2, 1], style="r--")

    log_df.plot(x='timestep', y='pyb_action3', ax=axs[2, 2])
    log_df.plot(x='timestep', y='pyb_action3_min', ax=axs[2, 2], style="r--")
    log_df.plot(x='timestep', y='pyb_action3_max', ax=axs[2, 2], style="r--")

    log_df.plot(x='timestep', y='pyb_action4', ax=axs[2, 3])
    log_df.plot(x='timestep', y='pyb_action4_min', ax=axs[2, 3], style="r--")
    log_df.plot(x='timestep', y='pyb_action4_max', ax=axs[2, 3], style="r--")

    log_df.plot(x='timestep', y='pyb_action5', ax=axs[2, 4])
    log_df.plot(x='timestep', y='pyb_action5_min', ax=axs[2, 4], style="r--")
    log_df.plot(x='timestep', y='pyb_action5_max', ax=axs[2, 4], style="r--")

    log_df.plot(x='timestep', y='pyb_action6', ax=axs[2, 5])
    log_df.plot(x='timestep', y='pyb_action6_min', ax=axs[2, 5], style="r--")
    log_df.plot(x='timestep', y='pyb_action6_max', ax=axs[2, 5], style="r--")

    log_df.plot(x='timestep', y='reward', ax=axs[3, 0], color="b", marker="x")
    log_df.plot(x='timestep', y='term1', ax=axs[3, 0], color="r")
    log_df.plot(x='timestep', y='term2', ax=axs[3, 0], color="g")

    log_df.plot(x='timestep', y='return', ax=axs[3, 1], color="m", marker="x")

    log_df.plot(x='timestep', y='distance', ax=axs[3, 2], color="b")

    log_df.plot(x='timestep', y='vel_dist', ax=axs[3, 3], color="g")
    ax_1 = axs[3, 3].twinx()
    log_df.plot(x='timestep', y='vel_pos', ax=ax_1, color="r")

    log_df.plot(x='timestep', y='acc_dist', ax=axs[3, 4], color="g")
    ax_2 = axs[3, 4].twinx()
    log_df.plot(x='timestep', y='acc_pos', ax=ax_2, color="r")

    log_df.plot(x='timestep', y='goal_x', ax=axs[3, 5], style='r')
    log_df.plot(x='timestep', y='goal_y', ax=axs[3, 5], style='b')
    log_df.plot(x='timestep', y='goal_z', ax=axs[3, 5], style='g')
    log_df.plot(x='timestep', y='tip_x', ax=axs[3, 5], style='xr')
    log_df.plot(x='timestep', y='tip_y', ax=axs[3, 5], style='xb')
    log_df.plot(x='timestep', y='tip_z', ax=axs[3, 5], style='xg')

    axs[0, 0].set_ylabel("joint1 pos (rad)")
    axs[0, 1].set_ylabel("joint2 pos (rad)")
    axs[0, 2].set_ylabel("joint3 pos (rad)")
    axs[0, 3].set_ylabel("joint4 pos (rad)")
    axs[0, 4].set_ylabel("joint5 pos (rad)")
    axs[0, 5].set_ylabel("joint6 pos (rad)")

    axs[1, 0].set_ylabel("Action 1")
    axs[1, 1].set_ylabel("Action 2")
    axs[1, 2].set_ylabel("Action 3")
    axs[1, 3].set_ylabel("Action 4)")
    axs[1, 4].set_ylabel("Action 5")
    axs[1, 5].set_ylabel("Action 6")

    axs[2, 0].set_ylabel("Pybullet action1")
    axs[2, 1].set_ylabel("Pybullet action2")
    axs[2, 2].set_ylabel("Pybullet action3")
    axs[2, 3].set_ylabel("Pybullet action4")
    axs[2, 4].set_ylabel("Pybullet action5")
    axs[2, 5].set_ylabel("Pybullet action6")

    axs[3, 0].set_ylabel("Reward")

    axs[3, 1].set_ylabel("Return")

    axs[3, 2].set_ylabel("Distance (m)")

    axs[3, 3].set_ylabel("Vel by distance change (m/s)", color="g")
    ax_1.set_ylabel("Vel by position changes (m/s)", color="r")
    axs[3, 3].tick_params(axis='y', labelcolor="g")
    ax_1.tick_params(axis='y', labelcolor="r")
    ax_1.legend(loc="upper right")
    axs[3, 3].legend(loc="lower right")

    axs[3, 4].set_ylabel("Acc by distance change (m/s^2)", color="g")
    ax_2.set_ylabel("Acc by position change (m/s^2)", color="r")
    axs[3, 4].tick_params(axis='y', labelcolor="g")
    ax_2.tick_params(axis='y', labelcolor="r")
    ax_2.legend(loc="upper right")
    axs[3, 4].legend(loc="lower right")

    axs[3, 5].set_ylabel("Coordinates (m)")
    axs[3, 5].legend(loc="upper right")
    # ax3.legend(bbox_to_anchor=(1, 1.05))
    # ax4.legend(bbox_to_anchor=(1.2, 1.05))

    plt.tight_layout()
    # plt.show()
    plt.savefig(SAVE_PATH, bbox_inches='tight')
