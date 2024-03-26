#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Using local machine for training"

# dataset
dataset=msrvtt
fps=3

DATA_PATH=/home/xianyang/Data/dataset/msrvtt_data
train_csv=${DATA_PATH}/MSRVTT_train.9k.csv
# training-9k
val_csv=dataset/msrvtt/MSRVTT_JSFUSION_test.csv
data_path=${DATA_PATH}/MSRVTT_data.json
features_path=${DATA_PATH}/resized_videos
pretrained_dir=/home/xianyang/Data/models/pretrained

# train or eval
do_train=1
do_eval=0


# learning strategies
pretrained_clip_name=ViT-B/32
lr=2e-3
coef_lr=1e-3
wd=0.2
epochs=5
optim=AdamW
max_words=32
max_frames=12
temperature_new=1.0
resume=None
#/home/xianyang/Data/code/DGL/pth_backup/msrvtt_dgl_transformer_vit32_45.8_ckpt.best.pth.tar
load_from_pretrained=0
batch_size=128           # single GPU batch size
batch_size_val=16
num_workers=8
n_display=50            # log per n_display
precision=amp

freeze_clip=1
time_embedding=0

shared_latent_space=transformer

# distributed training
init_method='tcp://127.0.0.1:6061'

model_dir=logs/${dataset}_dgl_${shared_latent_space}_epochs${epochs}_lr${lr}
echo "The model dir is ${model_dir}"

# CUDA_LAUNCH_BLOCKING=1
python  main.py \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --num_thread_reader ${num_workers} \
        --epochs ${epochs} \
        --batch_size ${batch_size} \
        --n_display ${n_display} \
        --train_csv ${train_csv} \
        --val_csv ${val_csv} \
        --data_path ${data_path} \
        --features_path ${features_path} \
        --output_dir ${model_dir} \
        --optim ${optim} \
        --lr ${lr} \
        --coef_lr ${coef_lr} \
        --wd ${wd} \
        --max_words ${max_words} \
        --max_frames ${max_frames} \
        --batch_size_val ${batch_size_val} \
        --datatype ${dataset} \
        --expand_msrvtt_sentences  \
        --feature_framerate ${fps} \
        --freeze_layer_num 12  \
        --slice_framepos 2 \
        --loose_type \
        --linear_patch 2d \
        --sim_header meanP \
        --pretrained_clip_name ${pretrained_clip_name} \
        --precision ${precision} \
        --init_method ${init_method} \
        --pretrained_dir ${pretrained_dir} \
        --freeze_clip ${freeze_clip} \
        --time_embedding ${time_embedding} \
        --resume ${resume} \
        --load_from_pretrained ${load_from_pretrained} \
        --shared_latent_space ${shared_latent_space}

done

echo "Training Finished!!!"

