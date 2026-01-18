#!/bin/bash
#SBATCH --job-name=icl
#SBATCH --qos=batch-short
#SBATCH --output=log_output/log_%A_%a.out
#SBATCH --error=log_error/log_%A_%a.err
#SBATCH --nodes=1 # Number of nodes required
#SBATCH --gres=gpu:1 # Number of GPUs required
#SBATCH --gpus-per-node=1 # Number of GPU per node
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH --cpus-per-task=1 # Number of CPUs per task
#SBATCH --mem=100G
#SBATCH --time 36:00:00
#SBATCH --partition=gpu-large
#SBATCH --sockets-per-node=1 # Number of sockets per node
#SBATCH --cores-per-socket=16 # Number of cores per socket
#SBATCH --constraint=gpu-h100

module load Anaconda3
source activate
conda activate timemoe

cd /home/s223540177/Time-MoE

export CUDA_VISIBLE_DEVICES=0

python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_ettm1 -d /scratch/s223540177/Time-Series-Library/data/all_datasets/ETT-small/ETTm1.csv -p 96 --data ETTm1 --enc_in 7 --dec_in 7 --c_out 7 --lam 0.01 --retrieval euclidean --num_closest_samples 18 --collapse_weight 1.0