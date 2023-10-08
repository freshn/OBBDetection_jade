#!/bin/bash -l
#SBATCH -J sl50 
#SBATCH --gres=gpu:4
#SBATCH -p big
#SBATCH -o ./output/sl50-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=2
#SBATCH --ntasks=4

module load cuda/10.2
module load python/anaconda3
source activate tov

export PYTHONPATH=./
./tools/dist_train.sh configs2/TinyPerson/base/sl_faster.py 2 --cfg-options optimizer.lr=0.01 data.workers_per_gpu=2 --work-dir work_dirs/sl50_faster/
