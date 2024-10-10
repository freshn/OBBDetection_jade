#!/bin/bash -l
#SBATCH -J test_obb_rcnn_ms_bh
#SBATCH --gres=gpu
#SBATCH -p big
#SBATCH -o ./output/test_obb_rcnn_ms_bh2x2-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --ntasks=1

module load cuda/10.2
module load python/anaconda3
source activate obbdet

export PYTHONPATH=./
# python tools/test.py configs/obb/oriented_rcnn/obb_rcnn_ms_bh.py \
#     work_dirs/obb_rcnn_ms_bh/latest.pth \
# --format-only --options save_dir=obb_rcnn_ms_bh 

python tools/test.py configs/obb/neg_copy/baseline_2x_bh.py \
    work_dirs/baseline_2x_bh/epoch_12.pth \
    --format-only --options save_dir=obb_rcnn_ms_bh2x2 