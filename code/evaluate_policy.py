""" Evaluate RL experiment and plot """


from pathlib import Path
import subprocess
import argparse
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp-id',
        help="Unique experiment ID",
        default=0,
        type=int)
    parser.add_argument(
        '--n-eval-steps',
        help="Number of evaluation timesteps",
        default=1000,
        type=int)
    parser.add_argument(
        '--log-info',
        help="Log information at each evaluation steps and save (0 or 1)",
        type=int,
        choices=[
            0,
            1],
        default=0)
    parser.add_argument(
        '--plot-dim',
        help="Plot end effector and goal position in real time (0: no plot, 2: 2D, 3: 3D)",
        type=int,
        default=0,
        choices=[
            0,
            2,
            3])
    parser.add_argument(
        '--render',
        help="1: Render environment, 0: don't render",
        type=int,
        choices=[
            0,
            1],
        default=0)
    args = parser.parse_args()

    EXP_ID = args.exp_id
    LOG_FOLDER = "logs/exp_" + str(EXP_ID)

    # Secondary arguments
    DETERMINISTIC = 1

    # Retrieve experiment parameters directly from log
    args_path = list(Path(LOG_FOLDER).rglob('args.yml'))[0]
    with open(args_path, 'r') as f:
        args_ordered = yaml.load(f, Loader=yaml.UnsafeLoader)
        args_dict = dict(args_ordered)

    ALGO = args_dict['algo']
    ENV_ID = args_dict['env']
    N_SEEDS = args_dict['n_seeds']

    # Evaluate single experiment
    for seed in range(N_SEEDS):
        print(
            "Evaluating env {} with algo {} and seed {}...".format(
                ENV_ID, ALGO, seed))

        args1 = [
            '--algo', ALGO,
            '--env', ENV_ID,
            '--n-eval-steps', args.n_eval_steps,
            '--log-folder', LOG_FOLDER,
            '--exp-id', seed + 1,
            '--deterministic', DETERMINISTIC,
            '--render', args.render,
            '--log-info', args.log_info,
            '--plot-dim', args.plot_dim
        ]
        args1 = list(map(str, args1))

        subprocess.call(["python", "enjoy.py"] + args1)

        LOG_FOLDER_SEED = LOG_FOLDER + "/" + ALGO + \
            "/" + ENV_ID + "_" + str(seed + 1) + "/"
        subprocess.call(["python",
                         "scripts/plot_training_1seed.py",
                         "--log-folder",
                         LOG_FOLDER_SEED,
                         "--env",
                         ENV_ID])

    # Plot results experiment
    args2 = [
        '--log-folder', LOG_FOLDER,
        '--env', ENV_ID,
        '--nb-seeds', N_SEEDS,
        '--n-eval-steps', args.n_eval_steps,
        '--deterministic-flag', 1,
        '--algo', ALGO,
        '--exp-id', EXP_ID
    ]

    args2 = list(map(str, args2))

    subprocess.call(["python", "scripts/plot_experiment.py"] + args2)

    if args.log_info:
        subprocess.call(["python",
                         "scripts/plot_episode_eval_log.py",
                         "--exp-id",
                         str(EXP_ID)])
