#!/bin/bash -l
#SBATCH -J test_obb_rcnn_ss_r
#SBATCH --gres=gpu
#SBATCH -p small
#SBATCH -o ./output/test_obb_rcnn_ss_r-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=1
#SBATCH --ntasks=1

module load cuda/10.2
module load python/anaconda3
source activate obbdet

export PYTHONPATH=./
python tools/test.py configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_dota10.py ckpt/obb_official_ss.pth \
--format-only --options save_dir=obb_rcnn_ss_r 

