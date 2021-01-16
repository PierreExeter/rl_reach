import argparse
import importlib
import os
import widowx_env
import time
import pandas as pd
import numpy as np
import torch as th
import yaml
import matplotlib.pyplot as plt
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecEnvWrapper
from utils import ALGOS, create_test_env, get_latest_run_id, get_saved_hyperparams
from utils.utils import calc_ep_success, calc_success_list, calc_reach_time, calc_mean_successratio_reachtime
from utils.utils import StoreDict
from collections import OrderedDict


def main():  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        help="environment ID",
        type=str,
        default="CartPole-v1")
    parser.add_argument(
        "-f",
        "--log-folder",
        help="Log folder",
        type=str,
        default="rl-trained-agents")
    parser.add_argument(
        "--algo",
        help="RL Algorithm",
        default="ppo",
        type=str,
        required=False,
        choices=list(ALGOS.keys()))
    parser.add_argument(
        "-n",
        "--n-eval-steps",
        help="Number of evaluation timesteps",
        default=1000,
        type=int)
    parser.add_argument(
        "--num-threads",
        help="Number of threads for PyTorch (-1 to use default)",
        default=-1,
        type=int)
    parser.add_argument(
        "--n-envs",
        help="number of environments",
        default=1,
        type=int)
    parser.add_argument(
        "--exp-id",
        help="Experiment ID (default: 0: latest, -1: no exp folder)",
        default=0,
        type=int)
    parser.add_argument(
        "--verbose",
        help="Verbose mode (0: no output, 1: INFO)",
        default=1,
        type=int)
    parser.add_argument(
        '--render',
        help="1: Render environment, 0: don't render",
        type=int,
        choices=[0, 1],
        default=0)
    parser.add_argument(
        '--deterministic',
        help="1: Use deterministic actions, 0: Use stochastic actions",
        type=int,
        choices=[0, 1],
        default=0)
    parser.add_argument(
        "--load-best",
        action="store_true",
        default=False,
        help="Load best model instead of last model if available")
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        default=False,
        help="Use stochastic actions (for DDPG/DQN/SAC)")
    parser.add_argument(
        "--norm-reward",
        action="store_true",
        default=False,
        help="Normalize reward if applicable (trained with VecNormalize)")
    parser.add_argument(
        "--seed",
        help="Random generator seed",
        type=int,
        default=0)
    parser.add_argument(
        "--reward-log",
        help="Where to log reward",
        default="",
        type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environemnt package modules to import (e.g. gym_minigrid)")
    parser.add_argument(
        "--env-kwargs",
        type=str,
        nargs="+",
        action=StoreDict,
        help="Optional keyword argument to pass to the env constructor")
    parser.add_argument(
        '--log-info',
        help="1: Log information at each evaluation steps and save, 0: don't log",
        type=int,
        choices=[0, 1],
        default=0)
    parser.add_argument(
        "--plot-dim",
        help="Plot end effector and goal position in real time (0: Don't plot, 2: 2D (default), 3: 3D)",
        type=int,
        default=0,
        choices=[0, 2, 3])
    args = parser.parse_args()

    #################################

    # Prepare log if needed
    if args.log_info:
        log_df = pd.DataFrame()
        log_dict = OrderedDict()

    # Prepare plot if needed
    if args.plot_dim == 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, figsize=(5, 10))
    elif args.plot_dim == 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')

    # Going through custom gym packages to let them register 
    # in the global registry
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    algo = args.algo
    folder = args.log_folder

    if args.exp_id == 0:
        args.exp_id = get_latest_run_id(os.path.join(folder, algo), env_id)
        print(f"Loading latest experiment, id={args.exp_id}")

    # Sanity checks
    if args.exp_id > 0:
        log_path = os.path.join(folder, algo, f"{env_id}_{args.exp_id}")
    else:
        log_path = os.path.join(folder, algo)

    assert os.path.isdir(log_path), f"The {log_path} folder was not found"

    found = False
    for ext in ["zip"]:
        model_path = os.path.join(log_path, f"{env_id}.{ext}")
        found = os.path.isfile(model_path)
        if found:
            break

    if args.load_best:
        model_path = os.path.join(log_path, "best_model.zip")
        found = os.path.isfile(model_path)

    if args.load_checkpoint is not None:
        model_path = os.path.join(
            log_path, f"rl_model_{args.load_checkpoint}_steps.zip")
        found = os.path.isfile(model_path)

    if not found:
        raise ValueError(
            f"No model found for {algo} on {env_id}, path: {model_path}")

    off_policy_algos = ["dqn", "ddpg", "sac", "her", "td3", "tqc"]

    if algo in off_policy_algos:
        args.n_envs = 1

    set_random_seed(args.seed)

    if args.num_threads > 0:
        if args.verbose > 1:
            print(f"Setting torch.num_threads to {args.num_threads}")
        th.set_num_threads(args.num_threads)

    stats_path = os.path.join(log_path, env_id)
    hyperparams, stats_path = get_saved_hyperparams(
        stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_id, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, "r") as f:
            # pytype: disable=module-attr
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_id,
        n_envs=args.n_envs,
        stats_path=stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=args.render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    kwargs = dict(seed=args.seed)
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))

    model = ALGOS[algo].load(model_path, env=env, **kwargs)

    obs = env.reset()

    # Force deterministic for DQN, DDPG, SAC and HER (that is a wrapper around)
    deterministic = args.deterministic or algo in off_policy_algos and not args.stochastic

    state = None
    episode_reward = 0.0
    episode_rewards, episode_lengths = [], []
    ep_len = 0
    successes = []  # For HER, monitor success rate

    episode_nb = 0
    success_threshold_50 = 0.05
    success_threshold_20 = 0.02
    success_threshold_10 = 0.01
    success_threshold_5 = 0.005
    success_threshold_2 = 0.002
    success_threshold_1 = 0.001
    success_threshold_05 = 0.0005
    ep_success_list_50 = []
    ep_success_list_20 = []
    ep_success_list_10 = []
    ep_success_list_5 = []
    ep_success_list_2 = []
    ep_success_list_1 = []
    ep_success_list_05 = []
    success_list_50 = []
    success_list_20 = []
    success_list_10 = []
    success_list_5 = []
    success_list_2 = []
    success_list_1 = []
    success_list_05 = []

    # Moved render flag outside the loop (Pierre)
    if args.render:
        env.render("human")

    for t in range(args.n_eval_steps):
        action, state = model.predict(
            obs, state=state, deterministic=deterministic)
        obs, reward, done, infos = env.step(action)

        # Slow down simulation when rendering (Pierre)
        if args.render:
            if "widowx" in env_id:
                time.sleep(1. / 30.)
            else:
                env.render()

        if "widowx" in env_id:
            # Update episode success list
            ep_success_list_50 = calc_ep_success(
                success_threshold_50, ep_success_list_50, infos)
            ep_success_list_20 = calc_ep_success(
                success_threshold_20, ep_success_list_20, infos)
            ep_success_list_10 = calc_ep_success(
                success_threshold_10, ep_success_list_10, infos)
            ep_success_list_5 = calc_ep_success(
                success_threshold_5, ep_success_list_5, infos)
            ep_success_list_2 = calc_ep_success(
                success_threshold_2, ep_success_list_2, infos)
            ep_success_list_1 = calc_ep_success(
                success_threshold_1, ep_success_list_1, infos)
            ep_success_list_05 = calc_ep_success(
                success_threshold_05, ep_success_list_05, infos)

        episode_reward += reward[0]
        ep_len += 1

        # Real time plot
        if args.plot_dim == 2:

            goal = infos[0]['goal_pos']
            tip = infos[0]['endeffector_pos']

            ax1.cla()
            ax1.plot(goal[0], goal[2], marker='o', color='g',
                     linestyle='', markersize=10, label="goal", alpha=0.5)
            ax1.plot(tip[0], tip[2], marker='x', color='r',
                     linestyle='', markersize=10, label="end effector", mew=3)

            circ_1_50 = plt.Circle(
                (goal[0],
                 goal[2]),
                radius=success_threshold_50,
                edgecolor='g',
                facecolor='w',
                linestyle='--',
                label="50 mm")
            circ_1_20 = plt.Circle(
                (goal[0],
                 goal[2]),
                radius=success_threshold_20,
                edgecolor='b',
                facecolor='w',
                linestyle='--',
                label="20 mm")
            circ_1_10 = plt.Circle(
                (goal[0],
                 goal[2]),
                radius=success_threshold_10,
                edgecolor='m',
                facecolor='w',
                linestyle='--',
                label="10 mm")
            circ_1_5 = plt.Circle(
                (goal[0],
                 goal[2]),
                radius=success_threshold_5,
                edgecolor='r',
                facecolor='w',
                linestyle='--',
                label="5 mm")
            ax1.add_patch(circ_1_50)
            ax1.add_patch(circ_1_20)
            ax1.add_patch(circ_1_10)
            ax1.add_patch(circ_1_5)

            ax1.set_xlim([-0.25, 0.25])
            ax1.set_ylim([0, 0.5])
            ax1.set_xlabel("x (m)", fontsize=15)
            ax1.set_ylabel("z (m)", fontsize=15)

            ax2.cla()
            ax2.plot(goal[1], goal[2], marker='o', color='g',
                     linestyle='', markersize=10, alpha=0.5)
            ax2.plot(
                tip[1],
                tip[2],
                marker='x',
                color='r',
                linestyle='',
                markersize=10,
                mew=3)

            circ_2_50 = plt.Circle(
                (goal[1],
                 goal[2]),
                radius=success_threshold_50,
                edgecolor='g',
                facecolor='w',
                linestyle='--')
            circ_2_20 = plt.Circle(
                (goal[1],
                 goal[2]),
                radius=success_threshold_20,
                edgecolor='b',
                facecolor='w',
                linestyle='--')
            circ_2_10 = plt.Circle(
                (goal[1],
                 goal[2]),
                radius=success_threshold_10,
                edgecolor='m',
                facecolor='w',
                linestyle='--')
            circ_2_5 = plt.Circle(
                (goal[1],
                 goal[2]),
                radius=success_threshold_5,
                edgecolor='r',
                facecolor='w',
                linestyle='--')
            ax2.add_patch(circ_2_50)
            ax2.add_patch(circ_2_20)
            ax2.add_patch(circ_2_10)
            ax2.add_patch(circ_2_5)

            ax2.set_xlim([-0.25, 0.25])
            ax2.set_ylim([0, 0.5])
            ax2.set_xlabel("y (m)", fontsize=15)
            ax2.set_ylabel("z (m)", fontsize=15)

            ax1.legend(loc='upper left', bbox_to_anchor=(
                0, 1.2), ncol=3, fancybox=True, shadow=True)

            fig.suptitle("timestep " + str(ep_len) + " | distance to target: " +
                         str(round(infos[0]['distance'] * 1000, 1)) + " mm")
            plt.pause(0.01)
            # plt.show()

        elif args.plot_dim == 3:

            goal = infos[0]['goal_pos']
            tip = infos[0]['endeffector_pos']

            ax.cla()
            ax.plot([goal[0]], [goal[1]], zs=[goal[2]], marker='o',
                    color='g', linestyle='', markersize=10, alpha=0.5)
            ax.plot([tip[0]], [tip[1]], zs=[tip[2]], marker='x',
                    color='r', linestyle='', markersize=10, mew=3)
            ax.set_xlim([-0.2, 0.2])
            ax.set_ylim([-0.2, 0.2])
            ax.set_zlim([0, 0.5])
            ax.set_xlabel("x (m)", fontsize=15)
            ax.set_ylabel("y (m)", fontsize=15)
            ax.set_zlabel("z (m)", fontsize=15)

            fig.suptitle("timestep " + str(ep_len) + " | distance to target: " +
                         str(round(infos[0]['distance'] * 1000, 1)) + " mm")
            plt.pause(0.01)
            # plt.show()

        if args.log_info:

            log_dict['episode'] = episode_nb
            log_dict['timestep'] = t
            log_dict['action_1'] = action[0][0]
            log_dict['action_2'] = action[0][1]
            log_dict['action_3'] = action[0][2]
            log_dict['action_4'] = action[0][3]
            log_dict['action_5'] = action[0][4]
            log_dict['action_6'] = action[0][5]
            log_dict['joint_pos1'] = infos[0]['joint_pos'][0]
            log_dict['joint_pos2'] = infos[0]['joint_pos'][1]
            log_dict['joint_pos3'] = infos[0]['joint_pos'][2]
            log_dict['joint_pos4'] = infos[0]['joint_pos'][3]
            log_dict['joint_pos5'] = infos[0]['joint_pos'][4]
            log_dict['joint_pos6'] = infos[0]['joint_pos'][5]
            log_dict['joint1_min'] = -3.1
            log_dict['joint1_max'] = 3.1
            log_dict['joint2_min'] = -1.6
            log_dict['joint2_max'] = 1.6
            log_dict['joint3_min'] = -1.6
            log_dict['joint3_max'] = 1.6
            log_dict['joint4_min'] = -1.8
            log_dict['joint4_max'] = 1.8
            log_dict['joint5_min'] = -3.1
            log_dict['joint5_max'] = 3.1
            log_dict['joint6_min'] = 0.0
            log_dict['joint6_max'] = 0.0
            log_dict['action_low1'] = env.action_space.low[0]
            log_dict['action_low2'] = env.action_space.low[1]
            log_dict['action_low3'] = env.action_space.low[2]
            log_dict['action_low4'] = env.action_space.low[3]
            log_dict['action_low5'] = env.action_space.low[4]
            log_dict['action_low6'] = env.action_space.low[5]
            log_dict['action_high1'] = env.action_space.high[0]
            log_dict['action_high2'] = env.action_space.high[1]
            log_dict['action_high3'] = env.action_space.high[2]
            log_dict['action_high4'] = env.action_space.high[3]
            log_dict['action_high5'] = env.action_space.high[4]
            log_dict['action_high6'] = env.action_space.high[5]
            log_dict['reward'] = reward[0]
            log_dict['return'] = episode_reward
            log_dict['distance'] = infos[0]['distance']
            log_dict['goal_x'] = infos[0]['goal_pos'][0]
            log_dict['goal_y'] = infos[0]['goal_pos'][1]
            log_dict['goal_z'] = infos[0]['goal_pos'][2]
            log_dict['tip_y'] = infos[0]['endeffector_pos'][1]
            log_dict['tip_x'] = infos[0]['endeffector_pos'][0]
            log_dict['tip_z'] = infos[0]['endeffector_pos'][2]
            log_dict['done'] = done[0]
            # log_dict['obs'] = obs
            # log_dict['obs_space_low'] = env.observation_space.low
            # log_dict['obs_space_high'] = env.observation_space.high

            log_df = log_df.append(log_dict, ignore_index=True)

        if args.n_envs == 1:

            if done and args.verbose > 0:
                # NOTE: for env using VecNormalize, the mean reward
                # is a normalized reward when `--norm_reward` flag is passed
                # print(f"Episode Reward: {episode_reward:.2f}") # commented by Pierre
                # print("Episode Length", ep_len)  # commented by Pierre
                episode_rewards.append(episode_reward)
                episode_lengths.append(ep_len)
                episode_nb += 1

                if "widowx" in env_id:
                    # append the last element of the episode success list when
                    # episode is done
                    success_list_50 = calc_success_list(
                        ep_success_list_50, success_list_50)
                    success_list_20 = calc_success_list(
                        ep_success_list_20, success_list_20)
                    success_list_10 = calc_success_list(
                        ep_success_list_10, success_list_10)
                    success_list_5 = calc_success_list(
                        ep_success_list_5, success_list_5)
                    success_list_2 = calc_success_list(
                        ep_success_list_2, success_list_2)
                    success_list_1 = calc_success_list(
                        ep_success_list_1, success_list_1)
                    success_list_05 = calc_success_list(
                        ep_success_list_05, success_list_05)

                    # If the episode is successful and it starts from an
                    # unsucessful step, calculate reach time
                    reachtime_list_50 = calc_reach_time(ep_success_list_50)
                    reachtime_list_20 = calc_reach_time(ep_success_list_20)
                    reachtime_list_10 = calc_reach_time(ep_success_list_10)
                    reachtime_list_5 = calc_reach_time(ep_success_list_5)
                    reachtime_list_2 = calc_reach_time(ep_success_list_2)
                    reachtime_list_1 = calc_reach_time(ep_success_list_1)
                    reachtime_list_05 = calc_reach_time(ep_success_list_05)

                if args.log_info:
                    log_df = log_df[log_dict.keys()]  # sort columns

                    # add estimated tip velocity and acceleration (according to
                    # the documentation, 1 timestep = 240 Hz)
                    log_df['est_vel'] = log_df['distance'].diff() * 240
                    log_df['est_vel'].loc[0] = 0    # initial velocity is 0
                    log_df['est_acc'] = log_df['est_vel'].diff() * 240
                    log_df['est_acc'].loc[0] = 0    # initial acceleration is 0

                    log_df.to_csv(
                        log_path +
                        "/res_episode_" +
                        str(episode_nb) +
                        ".csv",
                        index=False)  # slow
                    # log_df.to_pickle(log_path+"/res_episode_"+str(episode)+".pkl")
                    # # fast

                # Reset for the new episode
                episode_reward = 0.0
                ep_len = 0
                state = None
                ep_success_list_50 = []
                ep_success_list_20 = []
                ep_success_list_10 = []
                ep_success_list_5 = []
                ep_success_list_2 = []
                ep_success_list_1 = []
                ep_success_list_05 = []

            # Reset also when the goal is achieved when using HER
            if done and infos[0].get("is_success") is not None:
                if args.verbose > 1:
                    print("Success?", infos[0].get("is_success", False))
                # Alternatively, you can add a check to wait for the end of the
                # episode
                if done:
                    obs = env.reset()
                if infos[0].get("is_success") is not None:
                    successes.append(infos[0].get("is_success", False))
                    episode_reward, ep_len = 0.0, 0

    if args.verbose > 0 and len(successes) > 0:
        print(f"Success rate: {100 * np.mean(successes):.2f}%")

    if args.verbose > 0 and len(episode_lengths) > 0:
        print(
            f"Mean episode length: {np.mean(episode_lengths):.2f} +/- {np.std(episode_lengths):.2f}")

    if args.verbose > 0 and len(episode_rewards) > 0:
        print(
            f"Mean reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")

        if "widowx" in env_id:
            SR_mean_50, RT_mean_50 = calc_mean_successratio_reachtime(
                success_threshold_50, success_list_50, reachtime_list_50)
            SR_mean_20, RT_mean_20 = calc_mean_successratio_reachtime(
                success_threshold_20, success_list_20, reachtime_list_20)
            SR_mean_10, RT_mean_10 = calc_mean_successratio_reachtime(
                success_threshold_10, success_list_10, reachtime_list_10)
            SR_mean_5, RT_mean_5 = calc_mean_successratio_reachtime(
                success_threshold_5, success_list_5, reachtime_list_5)
            SR_mean_2, RT_mean_2 = calc_mean_successratio_reachtime(
                success_threshold_2, success_list_2, reachtime_list_2)
            SR_mean_1, RT_mean_1 = calc_mean_successratio_reachtime(
                success_threshold_1, success_list_1, reachtime_list_1)
            SR_mean_05, RT_mean_05 = calc_mean_successratio_reachtime(
                success_threshold_05, success_list_05, reachtime_list_05)

            # log metrics to stats.csv
            d = {
                "Eval mean reward": np.mean(episode_rewards),
                "Eval std": np.std(episode_rewards),
                "success ratio 50mm": SR_mean_50,
                "Average reach time 50mm": RT_mean_50,
                "success ratio 20mm": SR_mean_20,
                "Average reach time 20mm": RT_mean_20,
                "success ratio 10mm": SR_mean_10,
                "Average reach time 10mm": RT_mean_10,
                "success ratio 5mm": SR_mean_5,
                "Average reach time 5mm": RT_mean_5,
                "success ratio 2mm": SR_mean_2,
                "Average reach time 2mm": RT_mean_2,
                "success ratio 1mm": SR_mean_1,
                "Average reach time 1mm": RT_mean_1,
                "success ratio 0.5mm": SR_mean_05,
                "Average reach time 0.5mm": RT_mean_05
            }

            # print("path:", log_path)
            df = pd.DataFrame(d, index=[0])
            df.to_csv(log_path + "/stats.csv", index=False)

    # Workaround for https://github.com/openai/gym/issues/893
    if args.render:
        if args.n_envs == 1 and "Bullet" not in env_id and isinstance(
                env, VecEnv):
            # DummyVecEnv
            # Unwrap env
            while isinstance(env, VecEnvWrapper):
                env = env.venv
            if isinstance(env, DummyVecEnv):
                env.envs[0].env.close()
            else:
                env.close()
        else:
            # SubprocVecEnv
            env.close()


if __name__ == "__main__":
    main()
