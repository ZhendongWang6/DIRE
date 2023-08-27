#!/bin/bash
### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
eval "$(conda shell.bash hook)"
conda activate dire
EXP_NAME="lsun_adm"
DATASETS="lsun_adm"
DATASETS_TEST="lsun_adm"
python train.py --gpus 0 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST
