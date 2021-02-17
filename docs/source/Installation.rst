***************
Installation
***************

Local installation
==================

.. note::
    **Prerequisites:**
        * Python 3
        * Conda
        * CUDA (if GPU is used)

Clone the repository

.. code-block:: bash

    git clone https://github.com/PierreExeter/rl_reach.git && cd rl_reach/code/

Install and activate the Conda environment

.. code-block:: bash

    conda env create -f environment.yml
    conda activate rl_reach

.. note::
    This Conda environment assumes that you have CUDA 11.1 installed. 
    If you are using another version of CUDA, you will have to install 
    Pytorch manually as indicated `here <https://pytorch.org/get-started/locally/>`_.


Test the installation
---------------------

Manual tests

.. code-block:: bash

    python tests/manual/1_test_widowx_env.py
    python tests/manual/2_test_train.py
    python tests/manual/3_test_enjoy.py
    python tests/manual/4_test_pytorch.py

Automated tests

.. code-block:: bash
    
    pytest tests/auto/all_tests.py -v


Docker installation
===================

.. note::
    The GPU image requires `nvidia-docker <https://github.com/NVIDIA/nvidia-docker>`_.

Clone the repository

.. code-block:: bash

    git clone https://github.com/PierreExeter/rl_reach.git && cd rl_reach/

Pull the Docker image (CPU or GPU)

.. code-block:: bash

    # CPU
    docker pull rlreach/rlreach-cpu:latest
    # GPU
    docker pull rlreach/rlreach-gpu:latest

or build the images from the Dockerfiles

.. code-block:: bash

    # CPU
    docker build -t rlreach/rlreach-cpu:latest . -f docker/Dockerfile_cpu
    # GPU
    docker build -t rlreach/rlreach-gpu:latest . -f docker/Dockerfile_gpu


Test the Docker images
----------------------

.. code-block:: bash

    # CPU
    ./docker/run_docker_cpu.sh pytest tests/auto/all_tests.py -v
    # GPU
    ./docker/run_docker_gpu.sh pytest tests/auto/all_tests.py -v


CodeOcean
===================

A `reproducible capsule <https://codeocean.com/capsule/4112840/tree/>`_ is available on the `CodeOcean <https://codeocean.com/>`_ platform.
