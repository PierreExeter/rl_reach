""" launch single RL experiment """

import subprocess
import os
import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-id',
        help="Unique experiment ID",
        default=0,
        type=int)
    parser.add_argument('--algo', help="RL Algorithm", default="ppo", type=str)
    parser.add_argument(
        '--env',
        help="Training environment",
        default="widowx_reacher-v1",
        type=str)
    parser.add_argument(
        '--n-timesteps',
        help="Number of timesteps",
        default=3000,
        type=int)
    parser.add_argument(
        '--n-seeds',
        help="Number of seeds",
        default=5,
        type=int)
    args = parser.parse_args()

    # ALGO = "td3"
    # ENV_ID = "widowx_reacher-v1"
    # N_SEEDS = 2
    # N_TIMESTEPS = 2000
    # EXP_ID = 7

    # Secondary arguments
    EVAL_FREQ = -1
    N_EVAL_EPISODES = 10
    LOG_FOLDER = "logs/exp_" + str(args.exp_id)
    NUM_THREADS = 6

    # Check if experiment exist
    if os.path.isdir(LOG_FOLDER):
        answer = None
        while answer not in ("Y", "n"):
            answer = input(
                "This experiment ID already exist. Do you still want to launch the experiment ? [Y/n] ")
            if answer == "Y":
                break
            elif answer == "n":
                print("Aborting launch...")
                sys.exit()
            else:
                print("Please enter Y or n.")

    # Launch experiment
    for seed in range(args.n_seeds):

        print(
            "Exp #{}: Training env {} with algo {} and seed {}...".format(
                args.exp_id,
                args.env,
                args.algo,
                seed))

        args_train = [
            "--algo", args.algo,
            "--env", args.env,
            "--n-timesteps", args.n_timesteps,
            "--seed", seed,
            "--n-seeds", args.n_seeds,
            "--eval-episodes", N_EVAL_EPISODES,
            "--eval-freq", EVAL_FREQ,
            "--log-folder", LOG_FOLDER,
            "--num-threads", NUM_THREADS
        ]
        args_train = list(map(str, args_train))

        # Start training and redirect output to file
        with open("submission_logs/log_" + args.algo + "_0" + str(seed) + ".run", 'w') as f:
            subprocess.call(["python", "train.py"] + args_train, stdout=f)
