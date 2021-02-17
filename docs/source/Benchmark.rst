*********
Benchmark
*********

The performance metrics of a number of experiments can be compared.
The evaluation metrics, environment's variables, hyperparameters used during the training 
and parameters for evaluating the environments are logged for each experiments in the file 
``benchmark/benchmark_results.csv``. Evaluation metrics of selected experiments ID can be plotted 
with the script ``scripts/plot_benchmark.py``.

With the local installation
===========================

.. csv-table:: Usage
   :header:  Flag , Description , Type , Example 

   ``--exp-list``,	List of experiments to consider for plotting,	*list of int*,	26 27 28 29
   ``--col``,	Name of the hyperparameter for the X axis,	*str*,	n_timesteps

The arguments of the ``--col`` flag correspond to the column headings of the `benchmark file <https://github.com/PierreExeter/rl_reach/blob/master/code/benchmark/benchmark_results.csv>`_.

Example:

.. code-block:: bash

   python scripts/plot_benchmark.py --exp-list 26 27 28 29 --col n_timesteps

The plots are generated in the folder ``benchmark/plots/``. Here is an example of experiment benchmark plot:

.. image:: ../images/benchmark_plot.png

With Docker
===========

Benchmark plots can be generated using the Docker images.

.. code-block:: bash

   # CPU
   docker run -it --rm --network host --ipc=host --mount src=$(pwd),target=/root/rl_reach/,type=bind rlreach/rlreach-cpu:latest bash -c "python scripts/plot_benchmark.py --exp-list 26 27 28 29 --col n_timesteps"
   # GPU 
   docker run -it --rm --runtime=nvidia --network host --ipc=host --mount src=$(pwd),target=/root/rl_reach/,type=bind rlreach/rlreach-gpu:latest bash -c "python scripts/plot_benchmark.py --exp-list 26 27 28 29 --col n_timesteps"

A Shell script is provided for ease of usability.

.. code-block:: bash

   # CPU
   ./docker/run_docker_cpu.sh python scripts/plot_benchmark.py --exp-list 26 27 28 29 --col n_timesteps
   # GPU
   ./docker/run_docker_gpu.sh python scripts/plot_benchmark.py --exp-list 26 27 28 29 --col n_timesteps

