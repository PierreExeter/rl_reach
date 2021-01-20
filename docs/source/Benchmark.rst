**********
Benchmark
**********

The evaluation metrics, environment's variables, hyperparameters used during the training 
and parameters for evaluating the environments are logged for each experiments in the file 
``benchmark/benchmark_results.csv``. Evaluation metrics of selected experiments ID can be plotted 
with the script ``scripts/plot_benchmark.py``. The plots are generated in the folder ``benchmark/plots/``.


.. csv-table:: Usage
   :header:  Flag , Description , Type , Example 

   ``--exp-list``,	List of experiments to consider for plotting,	*list of int*,	26 27 28 29
   ``--col``,	Name of the hyperparameter for the X axis,	*str*,	n_timesteps

The arguments of the ``--col`` flag correspond to the column headings of the `benchmark file <https://github.com/PierreExeter/rl_reach/blob/master/benchmark/benchmark_results.csv>`_.

Example:

.. code-block:: bash

   python scripts/plot_benchmark.py --exp-list 26 27 28 29 --col n_timesteps