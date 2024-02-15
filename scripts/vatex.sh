#!/bin/bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYTHONUNBUFFERED="True"
export NCCL_DEBUG=INFO
#export CUDA_LAUNCH_BLOCKING=1
# hyperparameter
#echo -n "input the gpu (seperate by comma (,) ): "
#read gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Using local machine for training"
# dataset
dataset=vatex
fps=3

DATA_PATH=/dataset/vatex_resized
features_path=${DATA_PATH}/vatex_resized_videos
data_path=${DATA_PATH}
pretrained_dir=/models/pretrained
train_csv=${DATA_PATH}/MSRVTT_train.9k.csv
# lmdb_dataset=${DATA_PATH}/lmdb/msrvtt.lmdb
lmdb_dataset=None

# train or eval
do_train=1
do_eval=0

# learning strategies
pretrained_clip_name=ViT-B/32
lr=5e-3
coef_lr=1e-3
wd=0.2
epochs=10
optim=AdamW
max_frames=12
temperature_new=1.0
resume=None
load_from_pretrained=0
batch_size=64           # single GPU batch size
batch_size_val=16
num_workers=8
n_display=50            # log per n_display
precision=amp
# precision='fp32'
# precision='fp16'
freeze_clip=1

# distributed training
init_method='tcp://127.0.0.1:6061'
camoe_dsl=0
text_prompt_length=8
local_each_frame_prompt_length=4
global_visual_prompt_length=4
unified_prompt_layers=12

shared_latent_space=linear
model_dir=logs/${dataset}_linear_proj_vt_prompt_vit32_lr${lr}
echo "The model dir is ${model_dir}"
# CUDA_LAUNCH_BLOCKING=1

python main.py \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --num_thread_reader ${num_workers} \
        --epochs ${epochs} \
        --batch_size ${batch_size} \
        --n_display ${n_display} \
        --lmdb_dataset ${lmdb_dataset} \
        --train_csv ${train_csv} \
        --val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
        --data_path ${data_path} \
        --features_path ${features_path} \
        --output_dir ${model_dir} \
        --optim ${optim} \
        --lr ${lr} \
        --coef_lr ${coef_lr} \
        --wd ${wd} \
        --max_words 32 \
        --max_frames ${max_frames} \
        --batch_size_val ${batch_size_val} \
        --datatype ${dataset} \
        --expand_msrvtt_sentences  \
        --feature_framerate ${fps} \
        --freeze_layer_num 0  \
        --slice_framepos 2 \
        --loose_type \
        --linear_patch 2d \
        --sim_header meanP \
        --pretrained_clip_name ${pretrained_clip_name} \
        --precision ${precision} \
        --init_method ${init_method} \
        --pretrained_dir ${pretrained_dir} \
        --freeze_clip ${freeze_clip} \
        --resume ${resume} \
        --load_from_pretrained ${load_from_pretrained} \
        --text_prompt_length ${text_prompt_length} \
        --local_each_frame_prompt_length ${local_each_frame_prompt_length} \
        --global_visual_prompt_length ${global_visual_prompt_length} \
        --unified_prompt_layers ${unified_prompt_layers} 

done

echo "Training Finished!!!"