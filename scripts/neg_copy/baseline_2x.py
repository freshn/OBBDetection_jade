#!/bin/bash -l
#SBATCH -J baseline_2x
#SBATCH -p big
#SBATCH -o ./output/neg_copy/baseline_2x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-gpu=4
#SBATCH --ntasks=32

module load cuda/10.2
module load python/anaconda3
source activate obbdet

export PYTHONPATH=./
PORT=29808 ./tools/dist_train.sh configs/obb/neg_copy/baseline_2x.py 8 \
    --options optimizer.lr=0.04 --work-dir work_dirs/baseline_2x/
