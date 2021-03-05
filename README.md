# RL Reach

[![Build Status](https://travis-ci.com/PierreExeter/rl_reach.svg?branch=master)](https://travis-ci.com/PierreExeter/rl_reach)
[![Documentation Status](https://readthedocs.org/projects/rl-reach/badge/?version=latest)](https://rl-reach.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![pylint Score](/docs/images/train.svg)](https://www.pylint.org/)
[![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/4112840/tree/v1)

RL Reach is a platform for running reproducible reinforcement learning experiments. Training environments are provided to solve the reaching task with the WidowX MK-II robotic arm.
The Gym environments and training scripts are adapted from [Replab](https://github.com/bhyang/replab) and [Stable Baselines Zoo](https://github.com/DLR-RM/rl-baselines3-zoo), respectively.

![Alt text](/docs/images/widowx_env.gif?raw=true "The Widowx Gym environment in Pybullet")


## Documentation

Please read the [documentation](https://rl-reach.readthedocs.io/en/latest/) to get started with RL Reach. More details can be found in the associated [journal publication](https://www.sciencedirect.com/science/article/pii/S2665963821000099) or [ArXiv ePrint](https://arxiv.org/abs/2102.04916).

## Installation

### 1. Local installation

```bash
# Clone the repository
git clone https://github.com/PierreExeter/rl_reach.git && cd rl_reach/code/

# Install and activate the Conda environment
conda env create -f environment.yml
conda activate rl_reach
```

Note, this Conda environment assumes that you have CUDA 11.1 installed. If you are using another version of CUDA, you will have to install Pytorch manually as indicated [here](https://pytorch.org/get-started/locally/).

### 2. Docker install

Pull the Docker image (CPU or GPU)

```bash
docker pull rlreach/rlreach-cpu:latest
docker pull rlreach/rlreach-gpu:latest
```

or build image from Dockerfile

```bash
docker build -t rlreach/rlreach-cpu:latest . -f docker/Dockerfile_cpu
docker build -t rlreach/rlreach-gpu:latest . -f docker/Dockerfile_gpu
```

Run commands inside the docker container with `run_docker_cpu.sh` and `run_docker_gpu.sh`.

Example:
```bash
./docker/run_docker_cpu.sh python run_experiments.py --exp-id 999 --algo ppo --env widowx_reacher-v1 --n-timesteps 30000 --n-seeds 2
./docker/run_docker_cpu.sh python evaluate_policy.py --exp-id 999 --n-eval-steps 1000 --log-info 0 --plot-dim 0 --render 0
```

Note, the GPU image requires [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

## 3. CodeOcean

A [reproducible capsule](https://codeocean.com/capsule/4112840/tree/) is available on CodeOcean.


## Test the installation

Manual tests

```bash
python tests/manual/1_test_widowx_env.py
python tests/manual/2_test_train.py
python tests/manual/3_test_enjoy.py
python tests/manual/4_test_pytorch.py
```

Automated tests

```bash
pytest tests/auto/all_tests.py -v
```

## Train RL agents

RL experiments can be launched with the script `run_experiments.py`.

Usage:
|    Flag        |              Description                           |  Type   |    Example                        |
|----------------|----------------------------------------------------|---------|-----------------------------------|
|`--exp-id`      |Unique experiment ID                                | *int*   | 999                                |  
|`--algo`        |RL algorithm                                        | *str*   | a2c, ddpg, her, ppo, sac, td3     |
|`--env`         |Training environment ID                             | *str*   | widowx_reacher-v1                 |
|`--n-timesteps` |Number of training timesteps                        | *int*   | 10<sup>3</sup> to 10<sup>12</sup> | 
|`--n-seeds`     |Number of runs with different initialisation seeds  | *int*   | 2 to 10                           |


Example:
```bash
python run_experiments.py --exp-id 999 --algo ppo --env widowx_reacher-v1 --n-timesteps 10000 --n-seeds 3
```
A Bash script that launches multiple experiments is provided for convenience:
```bash
./run_all_exp.sh
```

## Evaluate policy and save results

Trained models can be evaluated and the results can be saved with the script `evaluate_policy.py`.

Usage:
|    Flag        |              Description                            |  Type   |    Example                              |
|----------------|-----------------------------------------------------|---------|-----------------------------------------|
|`--exp-id`      | Unique experiment ID                                | *int*   | 999                                      | 
|`--n-eval-steps`| Number of evaluation timesteps                      | *int*   | 1000                                    |
|`--log-info`    | Enable information logging at each evaluation steps | *bool*  | 0 (default) or 1                        |
|`--plot-dim`    | Live rendering of end-effector and goal positions   | *int*   | 0: do not plot (default), 2: 2D or 3: 3D| 
|`--render`      | Render environment during evaluation                | *bool*  | 0 (default) or 1                        |

Example:
```bash
python evaluate_policy.py --exp-id 999 --n-eval-steps 1000 --log-info 0 --plot-dim 0 --render 0
```

If `--log-info` was enabled during evaluation, it is possible to plot some useful information as shown in the plot below.
```bash
python scripts/plot_episode_eval_log.py --exp-id 999
```
The plots are generated in the associated experiment folder, e.g. `logs/exp_999/ppo/`.

Example of environment evaluation plot:

![Alt text](/docs/images/plot_episode_eval_log.png)

Example of experiment learning curves:

![Alt text](/docs/images/reward_vs_timesteps_smoothed.png)

## Benchmark

The evaluation metrics, environment's variables, hyperparameters used during the training and parameters for evaluating the environments are logged for each experiments in the file `benchmark/benchmark_results.csv`. Evaluation metrics of selected experiments ID can be plotted with the script `scripts/plot_benchmark.py`. The plots are generated in the folder `benchmark/plots/`.

Usage:
|    Flag     |              Description                            |  Type        |    Example                       |
|-------------|-----------------------------------------------------|--------------|----------------------------------|
|`--exp-list` | List of experiments to consider for plotting        | *list of int*| 26 27 28 29                      | 
|`--col`      | Name of the hyperparameter for the X axis, see column names [here](code/benchmark/benchmark_results.csv)   | *str*  | n_timesteps |

Example:
```bash
python scripts/plot_benchmark.py --exp-list 26 27 28 29 --col n_timesteps
```

Example of benchmark plot:

![Alt text](/docs/images/benchmark_plot.png)

## Optimise hyperparameters

Hyperparameters can be tuned automatically with the optimisation framework [Optuna](https://optuna.readthedocs.io/en/stable/) using the script `train.py -optimize`.

Usage:
|    Flag             |              Description                           |  Type   |    Example                        |
|---------------------|----------------------------------------------------|---------|-----------------------------------|
|`--algo`             |RL algorithm                                        | *str*   | a2c, ddpg, her, ppo, sac, td3     |
|`--env`              |Training environment ID                             | *str*   | widowx_reacher-v1                 |
|`--n-timesteps`      |Number of training timesteps                        | *int*   | 10<sup>3</sup> to 10<sup>12</sup> | 
|`--n-trials`         |Number of optimisation trials                       | *int*   | 2 to 100                          |
|`--n-jobs`           |Number of parallel jobs                             | *int*   | 2 to 16                           |
|`--sampler`          |Sampler for optimisation search                     | *str*   | random, tpe, skopt                |
|`--pruner`           |Pruner to kill unpromising trials early             | *str*   | halving, median, none             |
|`--n-startup-trials` |Number of trials before using optuna sampler        | *int*   | 2 to 10                           |
|`--n-evaluations`    |Number of episode to evaluate a trial               | *int*   | 10 to 20                          |
|`--log-folder`       |Log folder for the results                          | *str*   | logs/opti                         |

Example:
```bash
python train.py -optimize --algo ppo --env widowx_reacher-v1 --n-timesteps 100000 --n-trials 100 --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti
```

A Bash script that launches multiple hyperparameter optimisation runs is provided for convenience:
```bash
./opti_all.sh
```


## Clean all the results (Reset the repository)

It could be convenient to clean all the results and log files. Warning, this cannot be undone!

```bash
./cleanAll.sh
```

## Training environments

A number of custom Gym environments are available in the `gym_envs` directory. They simulate the WidowX MK-II robotic manipulator with the Pybullet physics engine. The objective is to bring the end-effector as close as possible to a target position.

Each implemented environment is described [here](code/gym_envs/widowx_env/envs_list.csv). The action, observation and reward functions are given in [this table](code/gym_envs/widowx_env/reward_observation_action_shapes/reward_observation_action.pdf). Some environment renderings can be found below.


| Name     | Reaching task      | Rendering              |
| ---------| -------------------| -----------------------|
| widowx_reacher-v26 | Fixed position, no orientation | 
![Alt text](/docs/images/fixed_pos_no_orient.gif?raw=true "Fixed position no orientation")|
| widowx_reacher-v28 | Random position, no orientation | 
![Alt text](/docs/images/rand_pos_no_orient.gif?raw=true "Random position no orientation")|
| widowx_reacher-v32 | Fixed position, fixed orientation | 
![Alt text](/docs/images/fixed_pos_fixed_orient.gif?raw=true "Fixed position fixed orientation")|
| widowx_reacher-v33 | Fixed position, random orientation | 
![Alt text](/docs/images/fixed_pos_rand_orient.gif?raw=true "Fixed position random orientation")|
| widowx_reacher-v34 | Moving position, no orientation | 
![Alt text](/docs/images/moving_pos_no_orient.gif?raw=true "Moving position no orientation")|



## Tested on

- Ubuntu 18.04
- Python 3.7.9
- Conda 4.9.2
- CUDA 11.1

## Citation

Please cite this work as:

```bash
@article{aumjaud2021a,
author = {Aumjaud, Pierre and McAuliffe, David and Rodriguez-Lera, Francisco J and Cardiff, Philip},
journal = {Software Impacts},
pages = {100061},
volume = {8},
title = {{rl{\_}reach: Reproducible reinforcement learning experiments for robotic reaching tasks}},
archivePrefix = {arXiv},
arxivId = {2102.04916},
doi = {https://doi.org/10.1016/j.simpa.2021.100061},
year = {2021}
}
```