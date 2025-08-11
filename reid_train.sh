#!/bin/bash
# root_dir={YOUR WORDING ROOT DIR}
root_dir=/home/xinl123/my_Workfiles/8.ReID
DATASET_NAME=RSTP-Reid

CUDA_VISIBLE_DEVICES=0 \
    python reid_train.py \
    --name RDE \
    --img_aug \
    --txt_aug \
    --batch_size 8 \
    --root_dir $root_dir \
    --output_dir experiments \
    --dataset_name $DATASET_NAME \
    --loss_names ReID  \
    --pretrain_choice ViT-B/32 \
    --log_period 1 \
    --num_epoch 20