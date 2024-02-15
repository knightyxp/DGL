export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Using local machine for training"

# dataset
dataset=activity
fps=3

data_dir=${HOME}/dataset1/data_activity
DATA_PATH=dataset/activitynet
data_path=${DATA_PATH}
features_path=${DATA_PATH}/act_resized
pretrained_dir=models/pretrained
# lmdb_dataset=${DATA_PATH}/lmdb/activity.lmdb
lmdb_dataset=None
train_csv=dataset/MSRVTT/msrvtt_data/MSRVTT_train.7k.csv


# train or eval
do_train=1
do_eval=0


# learning strategies
pretrained_clip_name=ViT-B/32
lr=1e-2
coef_lr=1e-3
wd=0.2
epochs=10
optim=AdamW
num_workers=6
# FOR DiDeMo and ActivityNet, use more words and video frames
max_words=64
max_frames=64

time_embedding=0
batch_size=16           # single GPU batch size
n_display=50            # log per n_display
precision=amp
freeze_clip=1

text_prompt_length=8
local_each_frame_prompt_length=4
global_visual_prompt_length=4
unified_prompt_layers=12

# distributed training
init_method='tcp://127.0.0.1:6061'
camoe_dsl=0
shared_latent_space=linear

model_dir=/logs/${dataset}_linear_proj_vt_prompt_vit32_lr${lr}
echo "The model dir is ${model_dir}"


python  main.py \
        --do_train ${do_train} \
        --do_eval ${do_eval} \
        --num_thread_reader ${num_workers} \
        --epochs ${epochs} \
        --batch_size ${batch_size} \
        --n_display ${n_display} \
        --data_dir ${data_dir} \
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
        --max_words ${max_words} \
        --max_frames ${max_frames} \
        --batch_size_val 16 \
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
        --text_prompt_length ${text_prompt_length} \
        --local_each_frame_prompt_length ${local_each_frame_prompt_length} \
        --global_visual_prompt_length ${global_visual_prompt_length} \
        --unified_prompt_layers ${unified_prompt_layers} 
done
echo "Finish Training"