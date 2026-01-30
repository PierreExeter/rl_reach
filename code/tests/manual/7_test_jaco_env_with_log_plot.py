""" Test env with random actions """

import time
import gymnasium as gym
import gym_envs
import subprocess
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt


env = gym.make('ReachingJaco-v7')
plot_filename = "plot_ReachingJaco-v7_rand.png"

env.render()
obs, info = env.reset()

log_df = pd.DataFrame()
log_dict = OrderedDict()
EPISODE_RETURN = 0

for t in range(100):
    action = env.action_space.sample()
    # action = [0, 0, -2, 0, 0, 0]
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    EPISODE_RETURN += reward

    print("action: ", action)
    print("obs: ", obs)
    print("reward: ", reward)
    print("done: ", done)
    print("info: ", info)
    print("timestep: ", t)


    log_dict['episode'] = 1
    log_dict['timestep'] = t
    log_dict['action1'] = info['action'][0]
    log_dict['action2'] = info['action'][1]
    log_dict['action3'] = info['action'][2]
    log_dict['action4'] = info['action'][3]
    log_dict['action5'] = info['action'][4]
    log_dict['action6'] = info['action'][5]
    log_dict['action1_min'] = info['action_min'][0]
    log_dict['action2_min'] = info['action_min'][1]
    log_dict['action3_min'] = info['action_min'][2]
    log_dict['action4_min'] = info['action_min'][3]
    log_dict['action5_min'] = info['action_min'][4]
    log_dict['action6_min'] = info['action_min'][5]
    log_dict['action1_max'] = info['action_max'][0]
    log_dict['action2_max'] = info['action_max'][1]
    log_dict['action3_max'] = info['action_max'][2]
    log_dict['action4_max'] = info['action_max'][3]
    log_dict['action5_max'] = info['action_max'][4]
    log_dict['action6_max'] = info['action_max'][5]
    log_dict['pyb_action1'] = info['pybullet_action'][0]
    log_dict['pyb_action2'] = info['pybullet_action'][1]
    log_dict['pyb_action3'] = info['pybullet_action'][2]
    log_dict['pyb_action4'] = info['pybullet_action'][3]
    log_dict['pyb_action5'] = info['pybullet_action'][4]
    log_dict['pyb_action6'] = info['pybullet_action'][5]
    log_dict['pyb_action1_min'] = info['pybullet_action_min'][0]
    log_dict['pyb_action2_min'] = info['pybullet_action_min'][1]
    log_dict['pyb_action3_min'] = info['pybullet_action_min'][2]
    log_dict['pyb_action4_min'] = info['pybullet_action_min'][3]
    log_dict['pyb_action5_min'] = info['pybullet_action_min'][4]
    log_dict['pyb_action6_min'] = info['pybullet_action_min'][5]
    log_dict['pyb_action1_max'] = info['pybullet_action_max'][0]
    log_dict['pyb_action2_max'] = info['pybullet_action_max'][1]
    log_dict['pyb_action3_max'] = info['pybullet_action_max'][2]
    log_dict['pyb_action4_max'] = info['pybullet_action_max'][3]
    log_dict['pyb_action5_max'] = info['pybullet_action_max'][4]
    log_dict['pyb_action6_max'] = info['pybullet_action_max'][5]
    log_dict['joint1_pos'] = info['joint_pos'][0]
    log_dict['joint2_pos'] = info['joint_pos'][1]
    log_dict['joint3_pos'] = info['joint_pos'][2]
    log_dict['joint4_pos'] = info['joint_pos'][3]
    log_dict['joint5_pos'] = info['joint_pos'][4]
    log_dict['joint6_pos'] = info['joint_pos'][5]
    log_dict['desired_joint1_pos'] = info['desired_joint_pos'][0]
    log_dict['desired_joint2_pos'] = info['desired_joint_pos'][1]
    log_dict['desired_joint3_pos'] = info['desired_joint_pos'][2]
    log_dict['desired_joint4_pos'] = info['desired_joint_pos'][3]
    log_dict['desired_joint5_pos'] = info['desired_joint_pos'][4]
    log_dict['desired_joint6_pos'] = info['desired_joint_pos'][5]
    log_dict['joint1_vel'] = info['joint_vel'][0]
    log_dict['joint2_vel'] = info['joint_vel'][1]
    log_dict['joint3_vel'] = info['joint_vel'][2]
    log_dict['joint4_vel'] = info['joint_vel'][3]
    log_dict['joint5_vel'] = info['joint_vel'][4]
    log_dict['joint6_vel'] = info['joint_vel'][5]
    log_dict['joint1_tor'] = info['joint_tor'][0]
    log_dict['joint2_tor'] = info['joint_tor'][1]
    log_dict['joint3_tor'] = info['joint_tor'][2]
    log_dict['joint4_tor'] = info['joint_tor'][3]
    log_dict['joint5_tor'] = info['joint_tor'][4]
    log_dict['joint6_tor'] = info['joint_tor'][5]
    log_dict['joint1_min'] = info['joint_min'][0]
    log_dict['joint2_min'] = info['joint_min'][1]
    log_dict['joint3_min'] = info['joint_min'][2]
    log_dict['joint4_min'] = info['joint_min'][3]
    log_dict['joint5_min'] = info['joint_min'][4]
    log_dict['joint6_min'] = info['joint_min'][5]
    log_dict['joint1_max'] = info['joint_max'][0]
    log_dict['joint2_max'] = info['joint_max'][1]
    log_dict['joint3_max'] = info['joint_max'][2]
    log_dict['joint4_max'] = info['joint_max'][3]
    log_dict['joint5_max'] = info['joint_max'][4]
    log_dict['joint6_max'] = info['joint_max'][5]
    log_dict['reward'] = reward
    log_dict['return'] = EPISODE_RETURN
    log_dict['distance'] = info['distance']
    log_dict['orientation'] = info['orientation']
    log_dict['goal_x'] = info['goal_pos'][0]
    log_dict['goal_y'] = info['goal_pos'][1]
    log_dict['goal_z'] = info['goal_pos'][2]
    log_dict['tip_x'] = info['endeffector_pos'][0]
    log_dict['tip_y'] = info['endeffector_pos'][1]
    log_dict['tip_z'] = info['endeffector_pos'][2]
    log_dict['goal_yaw'] = info['goal_orient'][0]
    log_dict['goal_pitch'] = info['goal_orient'][1]
    log_dict['goal_roll'] = info['goal_orient'][2]
    log_dict['tip_yaw'] = info['endeffector_orient'][0]
    log_dict['tip_pitch'] = info['endeffector_orient'][1]
    log_dict['tip_roll'] = info['endeffector_orient'][2]
    log_dict['done'] = done
    log_dict['term1'] = info['term1']
    log_dict['term2'] = info['term2']
    log_dict['vel_dist'] = info['vel_dist']
    log_dict['vel_pos'] = info['vel_pos']
    log_dict['collision'] = info['collision']

    # log_dict['obs'] = obs
    # log_dict['obs_space_low'] = env.observation_space.low
    # log_dict['obs_space_high'] = env.observation_space.high

    log_df = log_df.append(log_dict, ignore_index=True)



# sort columns
log_df = log_df[log_dict.keys()]

# add estimated tip velocity and acceleration
# (according to the Pybullet documentation, 1 timestep = 240 Hz)
log_df['acc_dist'] = log_df['vel_dist'].diff() * 240
log_df['acc_pos'] = log_df['vel_pos'].diff() * 240
# initial acceleration is 0
log_df['acc_dist'].loc[0] = (log_df['vel_dist'].loc[0] - 0 ) * 240
log_df['acc_pos'].loc[0] = (log_df['vel_pos'].loc[0] - 0 ) * 240

# Write to file
log_df.to_csv("res_episode.csv", index=False)  # slow

# # read data
# log_df = pd.read_csv(FILE_PATH)

# plot
fig, axs = plt.subplots(5, 6, figsize=(25, 10), dpi=300, sharex=True)

log_df.plot(x='timestep', y='joint1_pos', ax=axs[0, 0], style="gx")
log_df.plot(x='timestep', y='desired_joint1_pos', ax=axs[0, 0], style="b+")
log_df.plot(x='timestep', y='joint1_min', ax=axs[0, 0], style="r--")
log_df.plot(x='timestep', y='joint1_max', ax=axs[0, 0], style="r--")

log_df.plot(x='timestep', y='joint2_pos', ax=axs[0, 1], style="gx")
log_df.plot(x='timestep', y='desired_joint2_pos', ax=axs[0, 1], style="b+")
log_df.plot(x='timestep', y='joint2_min', ax=axs[0, 1], style="r--")
log_df.plot(x='timestep', y='joint2_max', ax=axs[0, 1], style="r--")

log_df.plot(x='timestep', y='joint3_pos', ax=axs[0, 2], style="gx")
log_df.plot(x='timestep', y='desired_joint3_pos', ax=axs[0, 2], style="b+")
log_df.plot(x='timestep', y='joint3_min', ax=axs[0, 2], style="r--")
log_df.plot(x='timestep', y='joint3_max', ax=axs[0, 2], style="r--")

log_df.plot(x='timestep', y='joint4_pos', ax=axs[0, 3], style="gx")
log_df.plot(x='timestep', y='desired_joint4_pos', ax=axs[0, 3], style="b+")
log_df.plot(x='timestep', y='joint4_min', ax=axs[0, 3], style="r--")
log_df.plot(x='timestep', y='joint4_max', ax=axs[0, 3], style="r--")

log_df.plot(x='timestep', y='joint5_pos', ax=axs[0, 4], style="gx")
log_df.plot(x='timestep', y='desired_joint5_pos', ax=axs[0, 4], style="b+")
log_df.plot(x='timestep', y='joint5_min', ax=axs[0, 4], style="r--")
log_df.plot(x='timestep', y='joint5_max', ax=axs[0, 4], style="r--")

log_df.plot(x='timestep', y='joint6_pos', ax=axs[0, 5], style="gx")
log_df.plot(x='timestep', y='desired_joint6_pos', ax=axs[0, 5], style="b+")
log_df.plot(x='timestep', y='joint6_min', ax=axs[0, 5], style="r--")
log_df.plot(x='timestep', y='joint6_max', ax=axs[0, 5], style="r--")

##################################################################

log_df.plot(x='timestep', y='joint1_vel', ax=axs[1, 0], color="b")
ax_joint1 = axs[1, 0].twinx()
log_df.plot(x='timestep', y='joint1_tor', ax=ax_joint1, color="g")

log_df.plot(x='timestep', y='joint2_vel', ax=axs[1, 1], color="b")
ax_joint2 = axs[1, 1].twinx()
log_df.plot(x='timestep', y='joint2_tor', ax=ax_joint2, color="g")

log_df.plot(x='timestep', y='joint3_vel', ax=axs[1, 2], color="b")
ax_joint3 = axs[1, 2].twinx()
log_df.plot(x='timestep', y='joint3_tor', ax=ax_joint3, color="g")

log_df.plot(x='timestep', y='joint4_vel', ax=axs[1, 3], color="b")
ax_joint4 = axs[1, 3].twinx()
log_df.plot(x='timestep', y='joint4_tor', ax=ax_joint4, color="g")

log_df.plot(x='timestep', y='joint5_vel', ax=axs[1, 4], color="b")
ax_joint5 = axs[1, 4].twinx()
log_df.plot(x='timestep', y='joint5_tor', ax=ax_joint5, color="g")

log_df.plot(x='timestep', y='joint6_vel', ax=axs[1, 5], color="b")
ax_joint6 = axs[1, 5].twinx()
log_df.plot(x='timestep', y='joint6_tor', ax=ax_joint6, color="g")

##################################################################

log_df.plot(x='timestep', y='action1', ax=axs[2, 0])
log_df.plot(x='timestep', y='action1_min', ax=axs[2, 0], style="r--")
log_df.plot(x='timestep', y='action1_max', ax=axs[2, 0], style="r--")

log_df.plot(x='timestep', y='action2', ax=axs[2, 1])
log_df.plot(x='timestep', y='action2_min', ax=axs[2, 1], style="r--")
log_df.plot(x='timestep', y='action2_max', ax=axs[2, 1], style="r--")

log_df.plot(x='timestep', y='action3', ax=axs[2, 2])
log_df.plot(x='timestep', y='action3_min', ax=axs[2, 2], style="r--")
log_df.plot(x='timestep', y='action3_max', ax=axs[2, 2], style="r--")

log_df.plot(x='timestep', y='action4', ax=axs[2, 3])
log_df.plot(x='timestep', y='action4_min', ax=axs[2, 3], style="r--")
log_df.plot(x='timestep', y='action4_max', ax=axs[2, 3], style="r--")

log_df.plot(x='timestep', y='action5', ax=axs[2, 4])
log_df.plot(x='timestep', y='action5_min', ax=axs[2, 4], style="r--")
log_df.plot(x='timestep', y='action5_max', ax=axs[2, 4], style="r--")

log_df.plot(x='timestep', y='action6', ax=axs[2, 5])
log_df.plot(x='timestep', y='action6_min', ax=axs[2, 5], style="r--")
log_df.plot(x='timestep', y='action6_max', ax=axs[2, 5], style="r--")

##################################################################

log_df.plot(x='timestep', y='pyb_action1', ax=axs[3, 0])
log_df.plot(x='timestep', y='pyb_action1_min', ax=axs[3, 0], style="r--")
log_df.plot(x='timestep', y='pyb_action1_max', ax=axs[3, 0], style="r--")

log_df.plot(x='timestep', y='pyb_action2', ax=axs[3, 1])
log_df.plot(x='timestep', y='pyb_action2_min', ax=axs[3, 1], style="r--")
log_df.plot(x='timestep', y='pyb_action2_max', ax=axs[3, 1], style="r--")

log_df.plot(x='timestep', y='pyb_action3', ax=axs[3, 2])
log_df.plot(x='timestep', y='pyb_action3_min', ax=axs[3, 2], style="r--")
log_df.plot(x='timestep', y='pyb_action3_max', ax=axs[3, 2], style="r--")

log_df.plot(x='timestep', y='pyb_action4', ax=axs[3, 3])
log_df.plot(x='timestep', y='pyb_action4_min', ax=axs[3, 3], style="r--")
log_df.plot(x='timestep', y='pyb_action4_max', ax=axs[3, 3], style="r--")

log_df.plot(x='timestep', y='pyb_action5', ax=axs[3, 4])
log_df.plot(x='timestep', y='pyb_action5_min', ax=axs[3, 4], style="r--")
log_df.plot(x='timestep', y='pyb_action5_max', ax=axs[3, 4], style="r--")

log_df.plot(x='timestep', y='pyb_action6', ax=axs[3, 5])
log_df.plot(x='timestep', y='pyb_action6_min', ax=axs[3, 5], style="r--")
log_df.plot(x='timestep', y='pyb_action6_max', ax=axs[3, 5], style="r--")

##################################################################

log_df.plot(x='timestep', y='reward', ax=axs[4, 0], color="b", marker="x")
log_df.plot(x='timestep', y='term1', ax=axs[4, 0], color="r")
log_df.plot(x='timestep', y='term2', ax=axs[4, 0], color="g")

log_df.plot(x='timestep', y='return', ax=axs[4, 1], color="m", marker="x")

log_df.plot(x='timestep', y='distance', ax=axs[4, 2], color="b")
ax_1 = axs[4, 2].twinx()
log_df.plot(x='timestep', y='orientation', ax=ax_1, color="r")

log_df.plot(x='timestep', y='vel_dist', ax=axs[4, 3], color="g")
log_df.plot(x='timestep', y='vel_pos', ax=axs[4, 3], color="g", marker="+")
ax_2 = axs[4, 3].twinx()
log_df.plot(x='timestep', y='acc_dist', ax=ax_2, color="r")
log_df.plot(x='timestep', y='acc_pos', ax=ax_2, color="r", marker="+")

log_df.plot(x='timestep', y='goal_x', ax=axs[4, 4], style='r')
log_df.plot(x='timestep', y='goal_y', ax=axs[4, 4], style='b')
log_df.plot(x='timestep', y='goal_z', ax=axs[4, 4], style='g')
log_df.plot(x='timestep', y='tip_x', ax=axs[4, 4], style='xr')
log_df.plot(x='timestep', y='tip_y', ax=axs[4, 4], style='xb')
log_df.plot(x='timestep', y='tip_z', ax=axs[4, 4], style='xg')

log_df.plot(x='timestep', y='goal_yaw', ax=axs[4, 5], style='r')
log_df.plot(x='timestep', y='goal_pitch', ax=axs[4, 5], style='b')
log_df.plot(x='timestep', y='goal_roll', ax=axs[4, 5], style='g')
log_df.plot(x='timestep', y='tip_yaw', ax=axs[4, 5], style='xr')
log_df.plot(x='timestep', y='tip_pitch', ax=axs[4, 5], style='xb')
log_df.plot(x='timestep', y='tip_roll', ax=axs[4, 5], style='xg')

##################################################################

axs[0, 0].set_ylabel("joint1 pos (rad)")
axs[0, 1].set_ylabel("joint2 pos (rad)")
axs[0, 2].set_ylabel("joint3 pos (rad)")
axs[0, 3].set_ylabel("joint4 pos (rad)")
axs[0, 4].set_ylabel("joint5 pos (rad)")
axs[0, 5].set_ylabel("joint6 pos (rad)")

##################################################################

axs[1, 0].set_ylabel("Velocity (rad/s)", color="b")
ax_joint1.set_ylabel("Torque (Nm)", color="g")
axs[1, 0].tick_params(axis='y', labelcolor="b")
ax_joint1.tick_params(axis='y', labelcolor="g")
ax_joint1.legend(loc="upper right")
axs[1, 0].legend(loc="upper left")

axs[1, 1].set_ylabel("Velocity (rad/s)", color="b")
ax_joint1.set_ylabel("Torque (Nm)", color="g")
axs[1, 1].tick_params(axis='y', labelcolor="b")
ax_joint1.tick_params(axis='y', labelcolor="g")
ax_joint1.legend(loc="upper right")
axs[1, 1].legend(loc="upper left")

axs[1, 2].set_ylabel("Velocity (rad/s)", color="b")
ax_joint1.set_ylabel("Torque (Nm)", color="g")
axs[1, 2].tick_params(axis='y', labelcolor="b")
ax_joint1.tick_params(axis='y', labelcolor="g")
ax_joint1.legend(loc="upper right")
axs[1, 2].legend(loc="upper left")

axs[1, 3].set_ylabel("Velocity (rad/s)", color="b")
ax_joint1.set_ylabel("Torque (Nm)", color="g")
axs[1, 3].tick_params(axis='y', labelcolor="b")
ax_joint1.tick_params(axis='y', labelcolor="g")
ax_joint1.legend(loc="upper right")
axs[1, 3].legend(loc="upper left")

axs[1, 4].set_ylabel("Velocity (rad/s)", color="b")
ax_joint1.set_ylabel("Torque (Nm)", color="g")
axs[1, 4].tick_params(axis='y', labelcolor="b")
ax_joint1.tick_params(axis='y', labelcolor="g")
ax_joint1.legend(loc="upper right")
axs[1, 4].legend(loc="upper left")

axs[1, 5].set_ylabel("Velocity (rad/s)", color="b")
ax_joint1.set_ylabel("Torque (Nm)", color="g")
axs[1, 5].tick_params(axis='y', labelcolor="b")
ax_joint1.tick_params(axis='y', labelcolor="g")
ax_joint1.legend(loc="upper right")
axs[1, 5].legend(loc="upper left")

##################################################################

axs[2, 0].set_ylabel("Action 1")
axs[2, 1].set_ylabel("Action 2")
axs[2, 2].set_ylabel("Action 3")
axs[2, 3].set_ylabel("Action 4")
axs[2, 4].set_ylabel("Action 5")
axs[2, 5].set_ylabel("Action 6")

##################################################################

axs[3, 0].set_ylabel("Pybullet action1")
axs[3, 1].set_ylabel("Pybullet action2")
axs[3, 2].set_ylabel("Pybullet action3")
axs[3, 3].set_ylabel("Pybullet action4")
axs[3, 4].set_ylabel("Pybullet action5")
axs[3, 5].set_ylabel("Pybullet action6")

##################################################################

axs[4, 0].set_ylabel("Reward")

axs[4, 1].set_ylabel("Return")

axs[4, 2].set_ylabel("Distance (m)", color="b")
ax_1.set_ylabel("Orientation", color="r")
axs[4, 2].tick_params(axis='y', labelcolor="b")
ax_1.tick_params(axis='y', labelcolor="r")
ax_1.legend(loc="upper right")
axs[4, 2].legend(loc="upper left")

axs[4, 3].set_ylabel("Velocity (m/s)", color="g")
ax_2.set_ylabel("Acceleration (m/s^2)", color="r")
axs[4, 3].tick_params(axis='y', labelcolor="g")
ax_2.tick_params(axis='y', labelcolor="r")
ax_2.legend(loc="upper right")
axs[4, 3].legend(loc="upper left")

axs[4, 4].set_ylabel("Coordinates (m)")
axs[4, 4].legend(loc="upper right")

axs[4, 5].set_ylabel("Orientation (rad)")
axs[4, 5].legend(loc="upper right")

# ax3.legend(bbox_to_anchor=(1, 1.05))
# ax4.legend(bbox_to_anchor=(1.2, 1.05))

##################################################################

plt.tight_layout()
# plt.show()
plt.savefig(plot_filename, bbox_inches='tight')
