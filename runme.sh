#!/bin/bash

#===============================================================================
# Configuration
#===============================================================================

# Dataset and workspace paths
DATASET_DIR="/DATA/pliu/EmotionData/GTZAN/genres_original/"
WORKSPACE="/home/pengliu/Private/panns_transfer_to_gtzan/"

# Model configuration - Transfer Learning with Pretrained PANNs
PRETRAINED_CHECKPOINT_PATH="/home/pengliu/Private/my_panns_transfer_to_gtzan/pretrained_model/Cnn6_mAP=0.343.pth"
MODEL_TYPE="FeatureAffectiveCnn6"
HOLDOUT_FOLD=1

# Training hyperparameters
LEARNING_RATE=1e-4
BATCH_SIZE=32
START_ITERATION=0
STOP_ITERATION=10000
GPU_ID=3  # Set to desired GPU ID

#===============================================================================
# Check Pretrained Model
#===============================================================================

echo "=== GTZAN Music Genre Classification with Transfer Learning ==="
echo "Dataset: 1000 audio files (700 train / 300 validation)"
echo "Model: AffectiveCnn6 with frozen visual system + 3-layer affective system"
echo "Evaluation: Audio-file-level (no data leakage)"
echo "----------------------------------------"

if [ -f "$PRETRAINED_CHECKPOINT_PATH" ]; then
    echo "✓ Pretrained model found: $PRETRAINED_CHECKPOINT_PATH"
    USE_PRETRAINED="--pretrained_checkpoint_path=$PRETRAINED_CHECKPOINT_PATH"
else
    echo "⚠ Pretrained model not found: $PRETRAINED_CHECKPOINT_PATH"
    echo "Training from scratch (no transfer learning)"
    USE_PRETRAINED=""
fi

#===============================================================================
# Feature Extraction (Full Dataset)
#===============================================================================

FEATURES_FILE="$WORKSPACE/features/features.h5"

# Option to force re-extraction with enhanced loading
FORCE_REEXTRACT=${FORCE_REEXTRACT:-false}

if [ "$FORCE_REEXTRACT" = "true" ] || [ ! -f "$FEATURES_FILE" ]; then
    if [ "$FORCE_REEXTRACT" = "true" ]; then
        echo "🔄 Force re-extracting features with enhanced loading..."
        rm -f "$FEATURES_FILE"
    else
        echo "Extracting features for full dataset (1000 files)..."
    fi
    
    python3 utils/features.py pack_audio_files_to_hdf5 \
        --dataset_dir=$DATASET_DIR \
        --workspace=$WORKSPACE
        # Enhanced loading with fallback methods for all 1000 files
else
    echo "Features file already exists: $FEATURES_FILE"
    echo "Skipping feature extraction..."
    echo "💡 To force re-extraction with enhanced loading: FORCE_REEXTRACT=true ./runme.sh"
fi

#===============================================================================
# Verify Dataset Completeness
#===============================================================================

echo "Verifying dataset completeness..."
python3 -c "
import h5py
import sys
try:
    with h5py.File('$FEATURES_FILE', 'r') as hf:
        audio_names = [name.decode() for name in hf['audio_name'][:]]
        unique_files = list(set(audio_names))
        print(f'✓ Found {len(unique_files)} unique audio files')
        if len(unique_files) < 900:
            print(f'⚠ WARNING: Expected ~1000 files, found only {len(unique_files)}')
            sys.exit(1)
        genres = {}
        for name in unique_files:
            genre = name.split('.')[0]
            genres[genre] = genres.get(genre, 0) + 1
        print(f'✓ Genres found: {list(genres.keys())}')
        if len(genres) < 10:
            print(f'⚠ WARNING: Expected 10 genres, found only {len(genres)}')
            print(f'Missing genres may cause validation issues')
except Exception as e:
    print(f'❌ Error reading features file: {e}')
    sys.exit(1)
"

#===============================================================================
# Create Index Files (Train/Validation Split)
#===============================================================================

echo "Creating train/validation index files with 70/30 split..."
python3 create_indexes.py

#===============================================================================
# Model Training (Full Dataset with Transfer Learning)
#===============================================================================

echo "Starting transfer learning training..."
echo "- Model: $MODEL_TYPE"
echo "- Transfer Learning: $([ -n "$USE_PRETRAINED" ] && echo "YES" || echo "NO")"
echo "- Learning Rate: $LEARNING_RATE"
echo "- Batch Size: $BATCH_SIZE"
echo "- Iterations: $STOP_ITERATION"
echo "- GPU: $GPU_ID"
echo "----------------------------------------"
echo "Note: Using full dataset (features.h5) - main.py default fixed"

CUDA_VISIBLE_DEVICES=$GPU_ID python3 pytorch/main.py train \
    --dataset_dir=$DATASET_DIR \
    --workspace=$WORKSPACE \
    --holdout_fold=$HOLDOUT_FOLD \
    --model_type=$MODEL_TYPE \
    $USE_PRETRAINED \
    --loss_type=clip_nll \
    --augmentation=none \
    --learning_rate=$LEARNING_RATE \
    --batch_size=$BATCH_SIZE \
    --resume_iteration=$START_ITERATION \
    --stop_iteration=$STOP_ITERATION \
    --cuda
    # Key improvements:
    # - Full dataset (1000 files) instead of mini_data (10 files)
    # - Proper 70/30 audio-file-level split (no data leakage)
    # - Changed augmentation from 'mixup' to 'none' (avoid dimension issues)
    # - Automatic index file creation
    # - Transfer learning with pretrained PANNs
    # - Fixed main.py default to use full dataset, not minidata

echo "Training completed!"
echo "Results saved in: $WORKSPACE"
echo "- Logs: $WORKSPACE/logs/"
echo "- Checkpoints: $WORKSPACE/checkpoints/"
echo "- Statistics: $WORKSPACE/statistics/"
 