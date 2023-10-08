#!/bin/bash -l
#SBATCH -J neg_copy8
#SBATCH -p big
#SBATCH -o ./output/neg_copy/neg_copy8-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-gpu=4
#SBATCH --ntasks=32

module load cuda/10.2
module load python/anaconda3
source activate obbdet

export PYTHONPATH=./
PORT=29808 ./tools/dist_train.sh configs/obb/neg_copy/neg_copy8.py 8 \
    --options optimizer.lr=0.04 --work-dir work_dirs/neg_copy8/
