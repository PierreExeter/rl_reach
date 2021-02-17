#!/bin/bash


# echo "A2C"
# python run_experiments.py --exp-id 1 --algo a2c --env widowx_reacher-v1 --n-timesteps 100000 --n-seeds 2
# echo "DDPG"
# python run_experiments.py --exp-id 2 --algo ddpg --env widowx_reacher-v1 --n-timesteps 100000 --n-seeds 2
# echo "PPO"
# python run_experiments.py --exp-id 3 --algo ppo --env widowx_reacher-v1 --n-timesteps 100000 --n-seeds 2
# echo "SAC"
# python run_experiments.py --exp-id 4 --algo sac --env widowx_reacher-v1 --n-timesteps 100000 --n-seeds 2
# echo "TD3"
# python run_experiments.py --exp-id 5 --algo td3 --env widowx_reacher-v1 --n-timesteps 100000 --n-seeds 2

# echo "A2C"
# python run_experiments.py --exp-id 6 --algo a2c --env widowx_reacher-v3 --n-timesteps 100000 --n-seeds 2
# echo "DDPG"
# python run_experiments.py --exp-id 7 --algo ddpg --env widowx_reacher-v3 --n-timesteps 100000 --n-seeds 2
# echo "PPO"
# python run_experiments.py --exp-id 8 --algo ppo --env widowx_reacher-v3 --n-timesteps 100000 --n-seeds 2
# echo "SAC"
# python run_experiments.py --exp-id 9 --algo sac --env widowx_reacher-v3 --n-timesteps 100000 --n-seeds 2
# echo "TD3"
# python run_experiments.py --exp-id 10 --algo td3 --env widowx_reacher-v3 --n-timesteps 100000 --n-seeds 2

# echo "HER"
# python run_experiments.py --exp-id 11 --algo her --env widowx_reacher-v2 --n-timesteps 100000 --n-seeds 2
# echo "HER"
# python run_experiments.py --exp-id 12 --algo her --env widowx_reacher-v4 --n-timesteps 100000 --n-seeds 2

echo "37"
python run_experiments.py --exp-id 39 --algo ppo --env widowx_reacher-v30 --n-timesteps 500000 --n-seeds 2
echo "38"
python run_experiments.py --exp-id 40 --algo ppo --env widowx_reacher-v31 --n-timesteps 500000 --n-seeds 2