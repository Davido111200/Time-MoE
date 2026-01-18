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
#SBATCH --time 24:00:00
#SBATCH --partition=gpu-large
#SBATCH --sockets-per-node=1 # Number of sockets per node
#SBATCH --cores-per-socket=16 # Number of cores per socket
#SBATCH --constraint=gpu-h100

module load Anaconda3
source activate
conda activate timemoe

cd /home/s223540177/Time-MoE

export CUDA_VISIBLE_DEVICES=0

python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_nn5 -d /scratch/s223540177/Time-Series-Library/data/all_datasets/nn5_daily_dataset_without_missing_values.csv -p 24 --data nn5 --enc_in 111 --dec_in 111 --c_out 111 --lam 0.01 --retrieval euclidean --num_closest_samples 12 --collapse_weight 0.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_nn5 -d /scratch/s223540177/Time-Series-Library/data/all_datasets/nn5_daily_dataset_without_missing_values.csv -p 36 --data nn5 --enc_in 111 --dec_in 111 --c_out 111 --lam 0.01 --retrieval euclidean --num_closest_samples 12 --collapse_weight 0.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_nn5 -d /scratch/s223540177/Time-Series-Library/data/all_datasets/nn5_daily_dataset_without_missing_values.csv -p 48 --data nn5 --enc_in 111 --dec_in 111 --c_out 111 --lam 0.01 --retrieval euclidean --num_closest_samples 12 --collapse_weight 0.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_nn5 -d /scratch/s223540177/Time-Series-Library/data/all_datasets/nn5_daily_dataset_without_missing_values.csv -p 60 --data nn5 --enc_in 111 --dec_in 111 --c_out 111 --lam 0.01 --retrieval euclidean --num_closest_samples 12 --collapse_weight 0.0 --batch_size 1024
