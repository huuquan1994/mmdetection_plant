#!/bin/bash

#$ -j y
#$ -cwd

# これはモジュールの読み込みのために必須
source /etc/profile.d/modules.sh

# module load python/3.6/3.6.5
module load cuda/10.0/10.0.130.1
module load cudnn/7.6/7.6.2
module load nccl/2.4/2.4.8-1

source ~/.bashrc
conda activate py36

./tools/dist_train.sh configs/aillis_cascade_rcnn_dconv_c3-c5_r101_fpn_1x.py 4 --validate