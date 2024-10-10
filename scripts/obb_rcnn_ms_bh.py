#!/bin/bash -l
#SBATCH -J obb_rcnn_ms_bh
#SBATCH --gres=gpu:4
#SBATCH -p big
#SBATCH -o ./output/obb_rcnn_ms_bh_bs2x2-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=4
#SBATCH --ntasks=8

module load cuda/10.2
module load python/anaconda3
# module load use.own use.dev gcc/5.4.0
source activate obbdet

export PYTHONPATH=./
# ./tools/dist_train.sh configs/obb/oriented_rcnn/obb_rcnn_ms_bh.py 1 \
#     --work-dir work_dirs/obb_rcnn_ms_bh/ 
#     # --options optimizer.lr=0.005 

./tools/dist_train.sh configs/obb/neg_copy/baseline_2x_bh.py 2 \
    --work-dir work_dirs/obb_rcnn_ms_bh \
    --options optimizer.lr=0.01 