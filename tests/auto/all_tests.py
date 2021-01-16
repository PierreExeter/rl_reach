import gym
import widowx_env
import pytest
import os
import shutil
import subprocess
from stable_baselines3.common.env_checker import check_env

######## TEST ENVIRONMENTS ###########


@pytest.mark.parametrize("env_id",
                         ['widowx_reacher-v1',
                          'widowx_reacher-v2',
                          'widowx_reacher-v3',
                          'widowx_reacher-v4'])
def test_envs(env_id):
    env = gym.make(env_id)
    # check_env(env)

    env.reset()

    for t in range(100):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)

    assert t == 99
    assert done

    env.close()


# doesn't work due to max_episode_steps=100 in __init__.py
# @pytest.mark.parametrize("env_id", ['widowx_reacher-v2', 'widowx_reacher-v4'])
# def test_goalEnvs(env_id):
#     env = gym.make(env_id)
#     assert env == isinstance(env, gym.GoalEnv)


######## TEST TRAIN AND ENJOY SCRIPTS ###########

# # Clean up: delete logs from previous test
log_folder = "logs/test/ppo/"
if os.path.isdir(log_folder):
    shutil.rmtree(log_folder)


def test_train():

    args = [
        "-n", 1000,
        "--algo", "ppo",
        "--env", 'CartPole-v1',
        "-f", log_folder
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'train.py'] + args)

    assert return_code == 0


def test_enjoy():

    args = [
        "-n", 300,
        "--algo", "ppo",
        "--env", 'CartPole-v1',
        "-f", log_folder,
        "--render", 0
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'enjoy.py'] + args)

    assert return_code == 0


######## TEST RUN EXPERIMENTS ###########


# Clean up: delete logs from previous test
for i in range(1001, 1005):
    log_folder = "logs/exp_" + str(i)
    if os.path.isdir(log_folder):
        shutil.rmtree(log_folder)


def test_exp1():

    args = [
        "--exp-id", 1001,
        "--algo", "a2c",
        "--env", 'widowx_reacher-v1',
        "--n-timesteps", 1000,
        "--n-seeds", 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'run_experiments.py'] + args)

    assert return_code == 0


def test_exp2():

    args = [
        "--exp-id", 1002,
        "--algo", "a2c",
        "--env", 'widowx_reacher-v3',
        "--n-timesteps", 1000,
        "--n-seeds", 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'run_experiments.py'] + args)

    assert return_code == 0


def test_exp3():

    args = [
        "--exp-id", 1003,
        "--algo", "her",
        "--env", 'widowx_reacher-v2',
        "--n-timesteps", 1000,
        "--n-seeds", 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'run_experiments.py'] + args)

    assert return_code == 0


def test_exp4():

    args = [
        "--exp-id", 1004,
        "--algo", "her",
        "--env", 'widowx_reacher-v4',
        "--n-timesteps", 1000,
        "--n-seeds", 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'run_experiments.py'] + args)

    assert return_code == 0


######## TEST EVAL POLICY ###########


def test_eval1():

    args = [
        '--exp-id', 1001,
        '--n-eval-steps', 100,
        '--log-info', 1,
        '--plot-dim', 0,
        '--render', 0
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'evaluate_policy.py'] + args)

    assert return_code == 0


def test_eval2():

    args = [
        '--exp-id', 1002,
        '--n-eval-steps', 100,
        '--log-info', 0,
        '--plot-dim', 2,
        '--render', 0
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'evaluate_policy.py'] + args)

    assert return_code == 0


def test_eval3():

    args = [
        '--exp-id', 1003,
        '--n-eval-steps', 100,
        '--log-info', 0,
        '--plot-dim', 3,
        '--render', 0
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'evaluate_policy.py'] + args)

    assert return_code == 0


def test_eval4():

    args = [
        '--exp-id', 1004,
        '--n-eval-steps', 100,
        '--log-info', 0,
        '--plot-dim', 0,
        '--render', 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'evaluate_policy.py'] + args)

    assert return_code == 0


######## TEST HYPERPARAMETER OPTIMISATION ###########

# # Clean up: delete logs from previous test
log_folder = "logs/test/opti/"
if os.path.isdir(log_folder):
    shutil.rmtree(log_folder)


def test_opti():

    args = [
        "-optimize",
        "--algo", "ppo",
        "--env", 'widowx_reacher-v1',
        "--n-timesteps", 2000,
        "--n-trials", 2,
        "--n-jobs", 8,
        "--sampler", "tpe",
        "--pruner", "median",
        "--n-startup-trials", 1,
        "--n-evaluations", 5,
        "--log-folder", log_folder
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'train.py'] + args)

    assert return_code == 0
