""" Automated test script with pytest """

import os
import shutil
import subprocess
import pytest
import gymnasium as gym
import gym_envs

######## TEST ENVIRONMENTS ###########


@pytest.mark.parametrize("env_id",
                         ['widowx_reacher-v1',
                          'widowx_reacher-v2',
                          'widowx_reacher-v3',
                          'widowx_reacher-v4',
                          'widowx_reacher-v5',
                          'widowx_reacher-v6',
                          'widowx_reacher-v7',
                          'widowx_reacher-v8',
                          'widowx_reacher-v9',
                          'widowx_reacher-v10',
                          'widowx_reacher-v11',
                          'widowx_reacher-v12',
                          'widowx_reacher-v13',
                          'widowx_reacher-v14',
                          'widowx_reacher-v15',
                          'widowx_reacher-v16',
                          'widowx_reacher-v17',
                          'widowx_reacher-v18',
                          'widowx_reacher-v19',
                          'widowx_reacher-v20',
                          'widowx_reacher-v21',
                          'widowx_reacher-v22',
                          'widowx_reacher-v23',
                          'widowx_reacher-v24',
                          'widowx_reacher-v25',
                          'widowx_reacher-v26',
                          'widowx_reacher-v27',
                          'widowx_reacher-v28',
                          'widowx_reacher-v29',
                          'widowx_reacher-v30',
                          'widowx_reacher-v31',
                          'widowx_reacher-v32',
                          'widowx_reacher-v33',
                          'widowx_reacher-v34',
                          'widowx_reacher-v35',
                          'widowx_reacher-v36',
                          'widowx_reacher-v37',
                          'widowx_reacher-v38'])
def test_envs(env_id):
    """ Test first Gym environments """
    env = gym.make(env_id)

    env.reset()

    for timestep in range(100):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)

    assert timestep == 99
    assert done

    env.close()


# doesn't work due to max_episode_steps=100 in __init__.py
# @pytest.mark.parametrize("env_id", ['widowx_reacher-v2', 'widowx_reacher-v4'])
# def test_goalEnvs(env_id):
#     env = gym.make(env_id)
#     assert env == isinstance(env, gym.GoalEnv)


######## TEST TRAIN AND ENJOY SCRIPTS ###########

LOG_FOLDER = "logs/test/"

def test_train():
    """ Test train.py script """

    args = [
        "-n", 1000,
        "--algo", "ppo",
        "--env", 'CartPole-v1',
        "-f", LOG_FOLDER
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'train.py'] + args)

    assert return_code == 0


def test_enjoy():
    """ Test enjoy script """

    args = [
        "-n", 300,
        "--algo", "ppo",
        "--env", 'CartPole-v1',
        "-f", LOG_FOLDER,
        "--render", 0
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'enjoy.py'] + args)

    assert return_code == 0


######## TEST RUN EXPERIMENTS ###########

def test_exp1():
    """ Test run_experiment script + fixed gym env """

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
    """ Test run_experiment script + fixed gym env """

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
    """ Test run_experiment script + random gym env """

    args = [
        "--exp-id", 1003,
        "--algo", "ppo",
        "--env", 'widowx_reacher-v2',
        "--n-timesteps", 1000,
        "--n-seeds", 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'run_experiments.py'] + args)

    assert return_code == 0


def test_exp4():
    """ Test run_experiment script + fixed goal env """

    args = [
        "--exp-id", 1004,
        "--algo", "ppo",
        "--env", 'widowx_reacher-v4',
        "--n-timesteps", 1000,
        "--n-seeds", 1
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'run_experiments.py'] + args)

    assert return_code == 0


######## TEST EVAL POLICY ###########


def test_eval1():
    """ Eval exp 1001 """

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
    """ Eval exp 1002 """

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
    """ Eval exp 1003 """

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
    """ Eval exp 1004 """

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

def test_opti():
    """ Test hyperparameter optimisation """

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
        "--log-folder", LOG_FOLDER
    ]

    args = list(map(str, args))

    return_code = subprocess.call(['python', 'train.py'] + args)

    assert return_code == 0


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup logs files once we are finished testing."""

    def remove_test_dir():
        log_dir = "logs/test/"
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)

        for i in range(1001, 1005):
            log_dir = "logs/exp_" + str(i)
            if os.path.isdir(log_dir):
                shutil.rmtree(log_dir)

    request.addfinalizer(remove_test_dir)
