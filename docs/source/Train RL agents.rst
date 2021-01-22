***************
Train RL agents
***************

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

With Docker:

.. code-block:: bash

   ./docker/run_docker_cpu.sh python run_experiments.py --exp-id 99 --algo ppo --env widowx_reacher-v1 --n-timesteps 10000 --n-seeds 2

The experiment's log files will generated in the working directory of the local machine.

A Bash script that launches multiple experiments is provided for convenience:

.. code-block:: bash

   ./run_all_exp.sh
