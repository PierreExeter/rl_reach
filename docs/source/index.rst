************************************
RL Reach Documentation
************************************

`RL Reach <https://github.com/PierreExeter/rl_reach>`_ is a toolbox for 
running reproducible reinforcement learning experiments applied 
to solving the reaching task with a robot manipulator.
The training Gym environments are adapted from 
the `Replab <https://github.com/bhyang/replab>`_ project.
The training scripts and RL algorithms are based on 
the `Stable Baselines 3 <https://github.com/DLR-RM/stable-baselines3>`_ implementation 
and its `Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ of trained agents.


.. image:: ../images/widowx_env.gif

Useful links
==================

* Github repository: `https://github.com/PierreExeter/rl_reach <https://github.com/PierreExeter/rl_reach>`_
* CodeOcean capsule: `https://codeocean.com/capsule/4112840/tree/ <https://codeocean.com/capsule/4112840/tree/>`_
* Software Impacts publication: `https://www.sciencedirect.com/science/article/pii/S2665963821000099 <https://www.sciencedirect.com/science/article/pii/S2665963821000099>`_
* ArXiv ePrint: `https://arxiv.org/abs/2102.04916 <https://arxiv.org/abs/2102.04916>`_
* Travis builds: `https://travis-ci.com/github/PierreExeter/rl_reach <https://travis-ci.com/github/PierreExeter/rl_reach>`_
* DockerHub CPU image: `https://hub.docker.com/r/rlreach/rlreach-cpu <https://hub.docker.com/r/rlreach/rlreach-cpu>`_
* DockerHub GPU image: `https://hub.docker.com/r/rlreach/rlreach-gpu <https://hub.docker.com/r/rlreach/rlreach-gpu>`_


Citation
==================

Please cite this work as follows:

.. code-block:: bibtex

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

Contents
==================

.. toctree::
   :maxdepth: 2

   Installation
   Train RL agents
   Evaluate policies
   Benchmark
   Optimise hyperparameters
   Training environments
   Documentation
