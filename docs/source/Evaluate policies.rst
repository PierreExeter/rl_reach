*************************
Evaluate trained policies
*************************

Trained models can be evaluated and the results can be saved with the script ``evaluate_policy.py``.

.. csv-table:: Usage
   :header:  Flag , Description , Type , Example 

   ``--exp-id``,	Unique experiment ID,	*int*,	99
   ``--n-eval-steps``,	Number of evaluation timesteps,	*int*,	1000
   ``--log-info``,	Enable information logging at each evaluation steps,	*bool*,	0 (default) or 1
   ``--plot-dim``,	Live rendering of end-effector and goal positions,	*int*,	0: do not plot (default), 2: 2D or 3: 3D
   ``--render``,	Render environment during evaluation,	*bool*,	0 (default) or 1

Example:

.. code-block:: bash

   python evaluate_policy.py --exp-id 99 --n-eval-steps 1000 --log-info 0 --plot-dim 0 --render 0

With Docker:

.. code-block:: bash

   ./docker/run_docker_cpu.sh python evaluate_policy.py --exp-id 99 --n-eval-steps 1000 --log-info 0 --plot-dim 0 --render 0

The experiment's log files will generated in the working directory of the local machine.

If ``--log-info`` was enabled during evaluation, it is possible to plot some useful information as shown in the plot below.

.. code-block:: bash

   python scripts/plot_episode_eval_log.py --exp-id 99

The plots are printed in the associated experiment folder, e.g. `logs/exp_99/ppo/`.

An example of environment evaluation plot:

.. image:: ../images/plot_episode_eval_log.png

An example of experiment learning curve:

.. image:: ../images/reward_vs_timesteps_smoothed.png
