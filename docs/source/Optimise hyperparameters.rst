*************************
Optimise hyperparameters
*************************

Hyperparameters can be tuned automatically with the optimisation 
framework `Optuna <https://optuna.readthedocs.io/en/stable/>`_ using
the script ``train.py -optimize``.

With the local installation
===========================

.. csv-table:: Usage
   :header:  Flag , Description , Type , Example 

   ``--algo``,	RL algorithm,	*str*,	"a2c, ddpg, her, ppo, sac, td3"
   ``--env``,	Training environment ID,	*str*,	widowx_reacher-v1
   ``--n-timesteps``,	Number of training timesteps,	*int*,	10\ :sup:`3` to 10\ :sup:`12`
   ``--n-trials``,	Number of optimisation trials,	*int*,	2 to 100
   ``--n-jobs``,	Number of parallel jobs,	*int*,	2 to 16
   ``--sampler``,	Sampler for optimisation search, *str*,	"random, tpe, skopt"
   ``--pruner``,	Pruner to kill unpromising trials early,	*str*,	"halving, median, none"
   ``--n-startup-trials``,	Number of trials before using optuna sampler,	*int*,	2 to 10
   ``--n-evaluations``,	Number of episode to evaluate a trial, int*,	10 to 20
   ``--log-folder``,	Log folder for the results,	*str*,	logs/opti

Example:

.. code-block:: bash

   python train.py -optimize --algo ppo --env widowx_reacher-v1 --n-timesteps 100000 --n-trials 100 --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti

A Bash script is provided that lanches multiple hyperparameter optimisation runs is provided for convenience.

.. code-block:: bash

   ./opti_all.sh


With Docker
===========

Hyperparameter optimisation can be carried out using the Docker images.

.. code-block:: bash

   # CPU
   docker run -it --rm --network host --ipc=host --mount src=$(pwd),target=/root/rl_reach/,type=bind rlreach/rlreach-cpu:latest bash -c "python train.py -optimize --algo ppo --env widowx_reacher-v1 --n-timesteps 100000 --n-trials 100 --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti"
   # GPU 
   docker run -it --rm --runtime=nvidia --network host --ipc=host --mount src=$(pwd),target=/root/rl_reach/,type=bind rlreach/rlreach-gpu:latest bash -c "python train.py -optimize --algo ppo --env widowx_reacher-v1 --n-timesteps 100000 --n-trials 100 --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti"

A Shell script is provided for ease of usability.

.. code-block:: bash

   # CPU
   ./docker/run_docker_cpu.sh python train.py -optimize --algo ppo --env widowx_reacher-v1 --n-timesteps 100000 --n-trials 100 --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti
   # GPU
   ./docker/run_docker_gpu.sh python train.py -optimize --algo ppo --env widowx_reacher-v1 --n-timesteps 100000 --n-trials 100 --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti
