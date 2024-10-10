#!/bin/bash -l
#SBATCH -J test_neg_copy8
#SBATCH --gres=gpu
#SBATCH -p big
#SBATCH -o ./output/neg_copy/test_neg_copy8-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --ntasks=1

module load cuda/10.2
module load python/anaconda3
source activate obbdet

export PYTHONPATH=./
python tools/test.py configs/obb/neg_copy/neg_copy8.py work_dirs/neg_copy8/epoch_24.pth \
--format-only --options save_dir=neg_copy8