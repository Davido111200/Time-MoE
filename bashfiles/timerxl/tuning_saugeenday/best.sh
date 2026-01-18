#!/bin/bash
#SBATCH --job-name=icl
#SBATCH --qos=batch-short
#SBATCH --output=log_output/log_%A_%a.out
#SBATCH --error=log_error/log_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time 24:00:00
#SBATCH --partition=gpu-large
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=16
#SBATCH --constraint=gpu-h100

module load Anaconda3
source activate
conda activate timemoe

cd /home/s223540177/Time-MoE

python /home/s223540177/Time-MoE/run_timer_xl_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/timer_finetuned_all_channels/saugeenday_dataset/epoch-1 -d /scratch/s223540177/Time-Series-Library/data/all_datasets/saugeenday_dataset.csv -p 96 --data saugeenday_dataset --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 13 --collapse_weight 1.0 --batch_size 1024 
python /home/s223540177/Time-MoE/run_timer_xl_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/timer_finetuned_all_channels/saugeenday_dataset/epoch-1 -d /scratch/s223540177/Time-Series-Library/data/all_datasets/saugeenday_dataset.csv -p 192 --data saugeenday_dataset --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 13 --collapse_weight 1.0 --batch_size 1024 
python /home/s223540177/Time-MoE/run_timer_xl_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/timer_finetuned_all_channels/saugeenday_dataset/epoch-1 -d /scratch/s223540177/Time-Series-Library/data/all_datasets/saugeenday_dataset.csv -p 336 --data saugeenday_dataset --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 13 --collapse_weight 1.0 --batch_size 1024 
python /home/s223540177/Time-MoE/run_timer_xl_unified_interpolate.py -m /scratch/s223540177/Time-MoE/checkpoints/timer_finetuned_all_channels/saugeenday_dataset/epoch-1 -d /scratch/s223540177/Time-Series-Library/data/all_datasets/saugeenday_dataset.csv -p 720 --data saugeenday_dataset --enc_in 1 --dec_in 1 --c_out 1 --lam 0.01 --retrieval euclidean --num_closest_samples 13 --collapse_weight 1.0 --batch_size 1024 