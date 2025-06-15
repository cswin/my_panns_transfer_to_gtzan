#!/bin/bash

#===============================================================================
# Configuration
#===============================================================================

# Dataset and workspace paths
DATASET_DIR="/DATA/pliu/EmotionData/GTZAN/genres_original/"
WORKSPACE="/home/pengliu/Private/panns_transfer_to_gtzan/"

# Model configuration
PRETRAINED_CHECKPOINT_PATH="/home/pengliu/Private/my_panns_transfer_to_gtzan/pretrained_model/Cnn6_mAP=0.343.pth"
MODEL_TYPE="Transfer_Cnn6"
HOLDOUT_FOLD=1

# Training hyperparameters
LEARNING_RATE=1e-4
BATCH_SIZE=32
START_ITERATION=0
STOP_ITERATION=10000
GPU_ID=3  # Set to desired GPU ID

#===============================================================================
# Feature Extraction
#===============================================================================

echo "Extracting features..."
python3 utils/features.py pack_audio_files_to_hdf5 \
    --dataset_dir=$DATASET_DIR \
    --workspace=$WORKSPACE \
    --mini_data

#===============================================================================
# Model Training
#===============================================================================

echo "Starting model training..."
CUDA_VISIBLE_DEVICES=$GPU_ID python3 pytorch/main.py train \
    --dataset_dir=$DATASET_DIR \
    --workspace=$WORKSPACE \
    --holdout_fold=$HOLDOUT_FOLD \
    --model_type=$MODEL_TYPE \
    --pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH \
    --loss_type=clip_nll \
    --augmentation='mixup' \
    --learning_rate=$LEARNING_RATE \
    --batch_size=$BATCH_SIZE \
    --resume_iteration=$START_ITERATION \
    --stop_iteration=$STOP_ITERATION \
    --cuda \
    --mini_data
 