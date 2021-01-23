***************
Train RL agents
***************

With the local installation
===========================

RL experiments can be launched with the script ``run_experiments.py``.

.. csv-table:: Usage
   :header: Flag, Description , Type , Example 

   ``--exp-id``, Unique experiment ID , *int* ,99
   ``--algo`` , RL algorithm , *str* , "a2c, ddpg, her, ppo, sac, td3"
   ``--env`` , Training environment ID , *str* , widowx_reacher-v1 
   ``--n-timesteps`` , Number of training timesteps , *int* , 10\ :sup:`3` to 10\ :sup:`12`
   ``--n-seeds`` , Number of runs with different initialisation seeds , *int* ,2 to 10

Example:

.. code-block:: bash

   python run_experiments.py --exp-id 99 --algo ppo --env widowx_reacher-v1 --n-timesteps 10000 --n-seeds 3

A Shell script that launches multiple experiments is provided for convenience:

.. code-block:: bash

   ./run_all_exp.sh

With Docker
===========

RL experiments can be launched from the Docker container (CPU or GPU version). The log files will generated in the working directory of the host machine.


.. code-block:: bash

   # CPU
   docker run -it --rm --network host --ipc=host --mount src=$(pwd),target=/root/rl_reach/,type=bind rlreach/rlreach-cpu:latest bash -c "python run_experiments.py --exp-id 99 --algo ppo --env widowx_reacher-v1 --n-timesteps 10000 --n-seeds 2"
   # GPU 
   docker run -it --rm --runtime=nvidia --network host --ipc=host --mount src=$(pwd),target=/root/rl_reach/,type=bind rlreach/rlreach-gpu:latest bash -c "python run_experiments.py --exp-id 99 --algo ppo --env widowx_reacher-v1 --n-timesteps 10000 --n-seeds 2"

A Shell script is provided for ease of usability.

.. code-block:: bash

   # CPU
   ./docker/run_docker_cpu.sh python run_experiments.py --exp-id 99 --algo ppo --env widowx_reacher-v1 --n-timesteps 10000 --n-seeds 2
   # GPU
   ./docker/run_docker_gpu.sh python run_experiments.py --exp-id 99 --algo ppo --env widowx_reacher-v1 --n-timesteps 10000 --n-seeds 2
