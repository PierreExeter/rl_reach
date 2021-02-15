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

echo "31"
python run_experiments.py --exp-id 31 --algo ppo --env widowx_reacher-v16 --n-timesteps 200000 --n-seeds 5
echo "32"
python run_experiments.py --exp-id 32 --algo ppo --env widowx_reacher-v17 --n-timesteps 200000 --n-seeds 5
echo "33"
python run_experiments.py --exp-id 33 --algo ppo --env widowx_reacher-v18 --n-timesteps 200000 --n-seeds 5
echo "34"
python run_experiments.py --exp-id 34 --algo ppo --env widowx_reacher-v19 --n-timesteps 1000000 --n-seeds 5
echo "35"
python run_experiments.py --exp-id 35 --algo ppo --env widowx_reacher-v26 --n-timesteps 1000000 --n-seeds 5
echo "36"
python run_experiments.py --exp-id 36 --algo ppo --env widowx_reacher-v27 --n-timesteps 1000000 --n-seeds 5