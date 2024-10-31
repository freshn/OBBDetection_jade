#!/bin/bash -l
#SBATCH -J eval_neg_copy8
#SBATCH -p devel
#SBATCH -o ./output/neg_copy/eval/eval_neg_copy8-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-gpu=2

module load python/anaconda3
module load cuda/10.2
source activate obbdet

export PYTHONPATH=./
tools/dist_test.sh configs/obb/neg_copy/neg_copy8.py \
    ./work_dirs/neg_copy8/epoch_24.pth 2 --out results/neg_copy8.pkl
    
python tools/confusion_matrix.py configs/obb/neg_copy/neg_copy8.py \
    results/neg_copy8.pkl matrix/neg_copy8/ --score-thr 0.6

# python tools/analysis_tools/analyze_results.py configs/copy_paste/neg_crop8_ada_ssp.py\
#     results/neg_copy8.pkl analysis/neg_copy8 --topk 30