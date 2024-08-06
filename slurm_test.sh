#!/bin/bash

#SBATCH -c8 --mem=25g
#SBATCH --gres=gpu:2
#SBATCH -p cs -q cspg

source /usr2/share/gpu.sbatch

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nohup python -m visdom.server -port 8097 &

python test.py --dataroot ./datasets/BCI \
  --name pix2pix_resnet_9blocks_PatchGAN_noclassifier_db \
  --model pix2pix \
  --pattern L1_L2_L3_L4 \
  --batch_size 1 \
  --init_type xavier \
  --num_threads 8 \
  --norm instance \
  --preprocess resize_and_crop \
  # --use_classification_wrapper False
