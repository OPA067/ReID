root_dir=/home/mytasks/reid
DATASET_NAME=RSTPReid

CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --batch_size 16 \
    --root_dir $root_dir \
    --output_dir experiments \
    --dataset_name $DATASET_NAME \
    --loss_names reid  \
    --pretrain_choice ViT-B/32 \
    --log_period 100 \
    --num_epoch 10 \