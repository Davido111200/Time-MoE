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
#SBATCH --time 4:00:00
#SBATCH --partition=gpu-large
#SBATCH --sockets-per-node=1 # Number of sockets per node
#SBATCH --cores-per-socket=8 # Number of cores per socket
#SBATCH --constraint=gpu-h100

module load Anaconda3
source activate
conda activate timemoe

cd /home/s223540177/Time-MoE

export CUDA_VISIBLE_DEVICES=0

# python -u run_eval.py \
#   -d /scratch/s223540177/Time-Series-Library/data/all_datasets/weather/weather.csv \
#   -p 96 \
#   -m /scratch/s223540177/Time-MoE/checkpoints/50M_weather \
#   --data weather \
#   --batch_size 1024

python -u run_eval.py \
  -d /scratch/s223540177/Time-Series-Library/data/all_datasets/weather/weather.csv \
  -p 192 \
  -m /scratch/s223540177/Time-MoE/checkpoints/50M_weather \
  --data weather \
  --batch_size 512
python -u run_eval.py \
  -d /scratch/s223540177/Time-Series-Library/data/all_datasets/weather/weather.csv \
  -p 336 \
  -m /scratch/s223540177/Time-MoE/checkpoints/50M_weather \
  --data weather \
  --batch_size 512
python -u run_eval.py \
  -d /scratch/s223540177/Time-Series-Library/data/all_datasets/weather/weather.csv \
  -p 720 \
  -m /scratch/s223540177/Time-MoE/checkpoints/50M_weather \
  --data weather \
  --batch_size 512
