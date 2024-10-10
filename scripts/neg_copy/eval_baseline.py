#!/bin/bash -l
#SBATCH -J eval_baseline
#SBATCH -p devel
#SBATCH -o ./output/neg_copy/eval/eval_baseline-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-gpu=2

module load python/anaconda3
module load cuda/10.2
source activate obbdet

export PYTHONPATH=./
PORT=29809 tools/dist_test.sh configs/obb/neg_copy/eval_baseline_2x.py \
    ./ckpt/obb_official.pth 2 --out results/baseline_2x.pkl
    
PORT=29810 python tools/confusion_matrix.py configs/obb/neg_copy/eval_baseline_2x.py \
    results/baseline_2x.pkl matrix/baseline_2x/ --score-thr 0.6

# python tools/analysis_tools/analyze_results.py configs/copy_paste/neg_crop8_ada_ssp_bh.py\
#     results/neg_copy8.pkl analysis/neg_copy8_bh --topk 30
