#!/bin/bash -l
#SBATCH -J obb_rcnn_ms_r
#SBATCH --gres=gpu
#SBATCH -p small
#SBATCH -o ./output/obb_rcnn_ms_r-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=4
#SBATCH --ntasks=4

module load cuda/10.2
module load python/anaconda3
source activate obbdet

export PYTHONPATH=./
./tools/dist_train.sh configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_ms_rr_dota10.py 1 --options optimizer.lr=0.005 --work-dir work_dirs/obb_rcnn_ms_r/
