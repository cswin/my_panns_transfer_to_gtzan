#!bin/bash

DATASET_DIR="/DATA/pliu/EmotionData/GTZAN/genres_original/"
WORKSPACE="/home/pengliu/Private/panns_transfer_to_gtzan/"

python3 utils/features.py pack_audio_files_to_hdf5 --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --mini_data

PRETRAINED_CHECKPOINT_PATH="/home/pengliu/Private/panns_transfer_to_gtzan/pretrained_model/Cnn6_mAP=0.343.pth"

CUDA_VISIBLE_DEVICES=3 python3 pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --holdout_fold=1 --model_type="Transfer_Cnn6" --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH --loss_type=clip_nll --augmentation='mixup' --learning_rate=1e-4 --batch_size=32 --resume_iteration=0 --stop_iteration=10000 --cuda
 