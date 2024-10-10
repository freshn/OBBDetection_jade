#!/bin/bash -l
#SBATCH -J obb_rcnn_ms_wave
#SBATCH --gres=gpu:4
#SBATCH -p big
#SBATCH -o ./output/obb_rcnn_ms_wave-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-gpu=4
#SBATCH --ntasks=16

module load cuda/10.2
module load python/anaconda3
# module load use.own use.dev gcc/5.4.0
source activate obbdet

export PYTHONPATH=./

# ./tools/dist_train.sh configs/obb/neg_copy/baseline_1x_wave.py 2 \
#     --work-dir work_dirs/obb_rcnn_ms_wave2x2/ \
#     --options optimizer.lr=0.01
    
python tools/train.py configs/obb/neg_copy/baseline_1x_wave.py \
    --work-dir work_dirs/obb_rcnn_ms_wave2x2/ \
    --options optimizer.lr=0.005