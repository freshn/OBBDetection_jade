#!/bin/bash -l
#SBATCH -J s2anet_bh
#SBATCH --gres=gpu:8
#SBATCH -p big
#SBATCH -o ./output/s2anet_bh-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=4
#SBATCH --ntasks=32

module load cuda/10.2
module load python/anaconda3
source activate obbdet

export PYTHONPATH=./
./tools/dist_train.sh configs/obb/s2anet/s2anet_bh.py 8 --options optimizer.lr=0.02 --work-dir work_dirs/s2anet_bh/
