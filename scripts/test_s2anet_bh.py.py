#!/bin/bash -l
#SBATCH -J test_s2anet_bh
#SBATCH --gres=gpu
#SBATCH -p big
#SBATCH -o ./output/test_s2anet_bh-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --ntasks=1

module load cuda/10.2
module load python/anaconda3
source activate obbdet

export PYTHONPATH=./
python tools/test.py configs/obb/s2anet/s2anet_bh.py work_dirs/s2anet_bh/latest.pth \
--format-only --options save_dir=s2anet_bh

