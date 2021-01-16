#!/bin/bash

echo "A2C"
python train.py --algo a2c --env widowx_reacher-v1 -n 100000 --n-trials 100 -optimize --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti &> submission_logs/log_a2c_opti.run
echo "DDPG"
python train.py --algo ddpg --env widowx_reacher-v1 -n 100000 --n-trials 100 -optimize --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti &> submission_logs/log_ddpg_opti.run
echo "PPO"
python train.py --algo ppo --env widowx_reacher-v1 -n 100000 --n-trials 100 -optimize --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti &> submission_logs/log_ppo_opti.run
echo "SAC"
python train.py --algo sac --env widowx_reacher-v1 -n 100000 --n-trials 100 -optimize --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti &> submission_logs/log_sac_opti.run
echo "TD3"
python train.py --algo td3 --env widowx_reacher-v1 -n 100000 --n-trials 100 -optimize --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti &> submission_logs/log_td3_opti.run


echo "A2C"
python train.py --algo a2c --env widowx_reacher-v3 -n 100000 --n-trials 100 -optimize --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti &> submission_logs/log_a2c_opti.run
echo "DDPG"
python train.py --algo ddpg --env widowx_reacher-v3 -n 100000 --n-trials 100 -optimize --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti &> submission_logs/log_ddpg_opti.run
echo "PPO"
python train.py --algo ppo --env widowx_reacher-v3 -n 100000 --n-trials 100 -optimize --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti &> submission_logs/log_ppo_opti.run
echo "SAC"
python train.py --algo sac --env widowx_reacher-v3 -n 100000 --n-trials 100 -optimize --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti &> submission_logs/log_sac_opti.run
echo "TD3"
python train.py --algo td3 --env widowx_reacher-v3 -n 100000 --n-trials 100 -optimize --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti &> submission_logs/log_td3_opti.run


echo "HER"
python train.py --algo her --env widowx_reacher-v2 -n 100000 --n-trials 100 -optimize --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti &> submission_logs/log_her_opti.run
echo "HER"
python train.py --algo her --env widowx_reacher-v4 -n 100000 --n-trials 100 -optimize --n-jobs 8 --sampler tpe --pruner median --n-startup-trials 10 --n-evaluations 10 --log-folder logs/opti &> submission_logs/log_her_opti.run

