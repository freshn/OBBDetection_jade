#!/bin/bash -l
#SBATCH -J vis_neg_copy8
#SBATCH --gres=gpu
#SBATCH -p small
#SBATCH -o ./output/neg_copy/vis_neg_copy-%j.out
#SBATCH --nodes=1

module load cuda/10.2
module load python/anaconda3
source activate obbdet

export PYTHONPATH=./
PORT=29237 python demo/huge_image_demo.py data/DOTA1_0/val/images/P0590.png \
        configs/obb/neg_copy/neg_copy8.py work_dirs/neg_copy8/epoch_24.pth \
        BboxToolkit/tools/split_configs/dota1_0/ms_val.json --out vis/P0590.png