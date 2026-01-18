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
# run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 1 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 2 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 3 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 4 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 5 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 6 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 7 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 8 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 9 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 10 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 11 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 12 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 13 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 14 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 15 --collapse_weight 1.0 --batch_size 1024
python run_eval_steering_finetuned_dtf_fullsteer_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/50M_us_births -d /scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv -p 96 --data us_births --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 16 --collapse_weight 1.0 --batch_size 1024
