#!/usr/bin/env bash

#GPU=0
#export CUDA_VISIBLE_DEVICES=$GPU

#python train.py \
#--name debug4 \
#--model object_detector_net_model_small \
#--checkpoints_dir ./checkpoints \
#--batch_size 30 \
#--gpu_ids 0 \
#--nepochs_no_decay 800 \
#--nepochs_decay 200

#python train.py \
#--name debug5 \
#--model object_detector_net_model_small \
#--checkpoints_dir ./checkpoints \
#--batch_size 30 \
#--gpu_ids 0 \
#--nepochs_no_decay 8000 \
#--nepochs_decay 2000


#GPU=1
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name debug6 \
#--model object_detector_net_model_small \
#--checkpoints_dir ./checkpoints \
#--batch_size 20 \
#--gpu_ids 0 \
#--nepochs_no_decay 8000 \
#--nepochs_decay 2000

#GPU=1
#export CUDA_VISIBLE_DEVICES=$GPU
#export CUDA_LAUNCH_BLOCKING=1
## no pretrained weights
#python train.py \
#--name debug11 \
#--model object_detector_net_model_small \
#--checkpoints_dir ./checkpoints \
#--batch_size 150 \
#--gpu_ids 0 \
#--nepochs_no_decay 10000 \
#--nepochs_decay 50

#GPU=0
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name prob_map9 \
#--model object_detector_net_prob_map \
#--checkpoints_dir ./checkpoints \
#--batch_size 120 \
#--gpu_ids 0 \
#--poses_g_sigma 0.12 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100

#GPU=1
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name prob_map10 \
#--model object_detector_net_prob_map \
#--checkpoints_dir ./checkpoints \
#--batch_size 120 \
#--gpu_ids 0 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100


#GPU=1
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name prob_map11 \
#--model object_detector_net_prob_map2 \
#--checkpoints_dir ./checkpoints \
#--batch_size 120 \
#--gpu_ids 0 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100
#
#GPU=0
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name prob_map12 \
#--model object_detector_net_prob_map \
#--checkpoints_dir ./checkpoints \
#--batch_size 120 \
#--gpu_ids 0 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100

#GPU=1
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name prob_map13 \
#--model object_detector_net_prob_map2 \
#--checkpoints_dir ./checkpoints \
#--batch_size 120 \
#--gpu_ids 0 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100

#GPU=0
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name prob_map14 \
#--model object_detector_net_prob_map2 \
#--checkpoints_dir ./checkpoints \
#--batch_size 120 \
#--gpu_ids 0 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100

#GPU=0
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name prob_map15 \
#--model object_detector_net_prob_map2 \
#--checkpoints_dir ./checkpoints \
#--batch_size 120 \
#--gpu_ids 0 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100

#GPU=1
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name prob_map16 \
#--model object_detector_net_prob_map2 \
#--checkpoints_dir ./checkpoints \
#--batch_size 120 \
#--gpu_ids 0 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100

#GPU=1
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name prob_map17 \
#--model object_detector_net_prob_map2 \
#--checkpoints_dir ./checkpoints \
#--batch_size 120 \
#--gpu_ids 0 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100

#GPU=1
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name object_detector_net_prob1 \
#--model object_detector_net_prob \
#--checkpoints_dir ./checkpoints \
#--batch_size 120 \
#--gpu_ids 0 \
#--lambda_prob 100 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100

##num_nc 16
#GPU=0
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name object_detector_net_prob2 \
#--model object_detector_net_prob \
#--checkpoints_dir ./checkpoints/object_detector_net_prob/ \
#--batch_size 120 \
#--gpu_ids 0 \
#--lambda_prob 100 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 400 \
#--nepochs_decay 200

##uv_prob_net2 32
#GPU=1
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name object_detector_net_prob3 \
#--model object_detector_net_prob \
#--checkpoints_dir ./checkpoints/object_detector_net_prob/ \
#--batch_size 120 \
#--gpu_ids 0 \
#--lambda_prob 100 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100

#uv_prob_net2 32
GPU=1
export CUDA_VISIBLE_DEVICES=$GPU
# no pretrained weights
python train.py \
--name object_detector_net_prob3 \
--model object_detector_net_prob \
--checkpoints_dir ./checkpoints/object_detector_net_prob/ \
--batch_size 120 \
--gpu_ids 0 \
--lambda_prob 100 \
--poses_g_sigma 0.6 \
--lr 0.0001 \
--nepochs_no_decay 800 \
--nepochs_decay 200


##3 with batchnorm
#GPU=0
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name object_detector_net_prob4 \
#--model object_detector_net_prob \
#--checkpoints_dir ./checkpoints/object_detector_net_prob/ \
#--batch_size 60 \
#--gpu_ids 0 \
#--lambda_prob 100 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100
#
##2 with mse
#GPU=0
#export CUDA_VISIBLE_DEVICES=$GPU
## no pretrained weights
#python train.py \
#--name object_detector_net_prob5 \
#--model object_detector_net_prob \
#--checkpoints_dir ./checkpoints/object_detector_net_prob/ \
#--batch_size 60 \
#--gpu_ids 0 \
#--lambda_prob 100 \
#--poses_g_sigma 0.6 \
#--nepochs_no_decay 500 \
#--nepochs_decay 100