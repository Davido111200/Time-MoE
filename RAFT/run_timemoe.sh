#!/bin/bash
#SBATCH --job-name timemoe
#SBATCH --output ./slurm/logs/timemoe.out # Output file name
#SBATCH --error ./slurm/logs/timemoe.err # Error log file name
## Below is for requesting the resource you want
#SBATCH --nodes=1 # Number of nodes required
## SBATCH --exclusive  --gres=gpu:1 # Number of GPUs required
#SBATCH --gres=gpu:1
## SBATCH --gpus-per-node=1 # Number of GPU per node
#SBATCH --ntasks-per-node=1 # Number of tasks per node
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=64gb
#SBATCH --time 3-00:00:00
#SBATCH --partition=gpu 
#SBATCH --sockets-per-node=1 # Number of sockets per node
#SBATCH --cores-per-socket=8 # Number of cores per socket
#SBATCH --qos=batch-short

## conda activate tsl
module load Anaconda3
source activate
conda activate raft
cd /scratch/s225250685/RAFT

bash scripts/timemoe/run_all.sh


