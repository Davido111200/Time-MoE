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

MODEL=/scratch/s223540177/Time-MoE/checkpoints/timer_finetuned_all_channels/us_births/epoch-1
DATA=/scratch/s223540177/Time-Series-Library/data/all_datasets/us_births_dataset.csv

LAMS="0.01 0.05 0.1"
WEIGHTS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"

for LAM in $LAMS; do
  for K in {1..16}; do
    for WEIGHT in $WEIGHTS; do
      echo "=== Running lam=${LAM}, num_closest_samples=${K}, weight=${WEIGHT} ==="
      python /home/s223540177/Time-MoE/run_timer_xl_unified_interpolate.py \
        -m "${MODEL}" \
        -d "${DATA}" \
        -p 96 \
        --data us_births \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --lam "${LAM}" \
        --retrieval euclidean \
        --num_closest_samples "${K}" \
        --collapse_weight "${WEIGHT}" \
        --batch_size 1024 \
        --pool_number 10000
    done
  done
done

