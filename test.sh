#!/bin/bash
### make sure that you have modified the EXP_NAME, CKPT, DATASETS_TEST
eval "$(conda shell.bash hook)"
conda activate dire

EXP_NAME="lsun_adm_release"
CKPT="/data3/wangzd/MSRA/result/guofeng-Midv5-450total-2e-6-rank64/1400/lsun_adm.pth"
DATASETS_TEST="release"
python test.py --gpus 0 --ckpt $CKPT --exp_name $EXP_NAME datasets_test $DATASETS_TEST