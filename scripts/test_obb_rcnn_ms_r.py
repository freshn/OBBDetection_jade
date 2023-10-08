#!/bin/bash -l
#SBATCH -J test_obb_rcnn_ms_r
#SBATCH --gres=gpu
#SBATCH -p big
#SBATCH -o ./output/test_obb_rcnn_ms_r-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --ntasks=1

module load cuda/10.2
module load python/anaconda3
source activate obbdet

export PYTHONPATH=./
python tools/test.py configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py work_dirs/obb_rcnn_ms_r/epoch_12.pth \
--format-only --options save_dir=obb_rcnn_ms_r_own_train

