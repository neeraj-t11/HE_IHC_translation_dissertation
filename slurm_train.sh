#!/bin/bash

#SBATCH -c8 --mem=25g
#SBATCH --gres=gpu:2
#SBATCH -p cs -q cspg

source /usr2/share/gpu.sbatch

# conda install visdom==0.1.8.8 dominate==2.4.0
# conda install visdom
# Verify that cv2 is installed
# python -c "import cv2; print('cv2 version:', cv2.__version__)"

python -c "import torch; torch.cuda.empty_cache()"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# echo $SLURMD_NODENAME > node_info.txt
nohup python -m visdom.server -port 8097 &
# torchrun train.py --dataroot ./datasets/BCI --pattern L1_L2_L3

# made batch size as 1 here, as gradient accumulation is applied with 4 accumulation steps
python train.py \
  --dataroot ./datasets/BCI \
  --pattern L1_L2_L3_L4 \
  --model pix2pix \
  --name pix2pix_resnet_9blocks_PatchGAN_classifierfm_db \
  --batch_size 1 \
  --init_type xavier \
  --num_threads 8 \
  --norm instance \
  --preprocess resize_and_crop \
  --use_classification_wrapper
   # --netG attention_unet_32 \
   