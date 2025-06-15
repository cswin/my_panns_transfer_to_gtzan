#!/bin/bash

# Script to run emotion regression training on Emo-Soundscapes dataset
# This script demonstrates how to:
# 1. Extract features from Emo-Soundscapes audio files
# 2. Train a PANNs model for valence/arousal prediction
#
# Usage:
#   bash run_emotion.sh                    # Run full pipeline (extract + train + eval)
#   bash run_emotion.sh --skip-extraction  # Skip feature extraction, use existing features

# =============================================================================
# Configuration - MODIFY THESE PATHS TO MATCH YOUR SETUP
# =============================================================================

# Parse command line arguments
SKIP_EXTRACTION=false
for arg in "$@"; do
    case $arg in
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Path to your Emo-Soundscapes dataset (containing audio folders)
EMO_AUDIO_DIR="/DATA/pliu/EmotionData/Emo-Soundscapes/audio_flat"

# Path to ratings directory (containing Valence.csv and Arousal.csv)
EMO_RATINGS_DIR="/DATA/pliu/EmotionData/Emo-Soundscapes/Emo-Soundscapes-Ratings"

# Your workspace directory
WORKSPACE="workspaces/emotion_regression"

# Feature file location (update this if features are in a different location)
FEATURE_FILE="features/emotion_features/emotion_features.h5"

# Check if we're using workspace-relative features
if [ "$SKIP_EXTRACTION" = false ]; then
    # If extracting, features will be in workspace
    FEATURE_FILE="$WORKSPACE/features/emotion_features.h5"
fi

# Path to pretrained PANNs model (download from PANNs repository)
PRETRAINED_MODEL="pretrained_model/Cnn6_mAP=0.343.pth"  # Cnn6 model
# PRETRAINED_MODEL="pretrained_model/Cnn14_mAP=0.431.pth"  # Cnn14 model (alternative)

# =============================================================================
# Validation Checks
# =============================================================================

echo "Validating setup..."

# Check if audio directory exists
if [ ! -d "$EMO_AUDIO_DIR" ]; then
    echo "Error: Audio directory not found: $EMO_AUDIO_DIR"
    echo "Please update EMO_AUDIO_DIR in the script to point to your flattened audio directory"
    exit 1
fi

# Check if ratings directory exists
if [ ! -d "$EMO_RATINGS_DIR" ]; then
    echo "Error: Ratings directory not found: $EMO_RATINGS_DIR"
    echo "Please update EMO_RATINGS_DIR in the script to point to your ratings directory"
    exit 1
fi

# Check if pretrained model exists
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "Error: Pretrained model not found: $PRETRAINED_MODEL"
    echo "Please download the Cnn6 model from PANNs repository and update PRETRAINED_MODEL path"
    exit 1
fi

# Create workspace directory if it doesn't exist
mkdir -p "$WORKSPACE"
mkdir -p "$WORKSPACE/checkpoints"
mkdir -p "$WORKSPACE/logs"
mkdir -p "$WORKSPACE/statistics"

echo "Setup validation passed!"

# =============================================================================
# Feature Extraction
# =============================================================================

if [ "$SKIP_EXTRACTION" = true ]; then
    echo "Skipping feature extraction (--skip-extraction flag provided)"
    
    # Check if features already exist
    if [ ! -f "$FEATURE_FILE" ]; then
        echo "Error: No existing features found at $FEATURE_FILE"
        echo "Either run without --skip-extraction or ensure features exist at the expected location"
        exit 1
    fi
    
    echo "Using existing features at $FEATURE_FILE"
    echo "⚠️  WARNING: If you're using old features (not segmented), you may need to re-extract!"
    echo "   The updated system expects 6 segments per audio file (7278 total samples)"
    echo "   Old format has 1 sample per audio file (1213 total samples)"
else
    echo "Step 1: Extracting features from Emo-Soundscapes dataset..."

    python extract_emotion_features.py \
        --audio_dir "$EMO_AUDIO_DIR" \
        --ratings_dir "$EMO_RATINGS_DIR" \
        --output_dir "$WORKSPACE/features"

    # Check if feature extraction was successful
    if [ ! -f "$FEATURE_FILE" ]; then
        echo "Error: Feature extraction failed! Expected features at: $FEATURE_FILE"
        exit 1
    fi

    echo "Feature extraction completed successfully!"
fi

# =============================================================================
# Data Validation
# =============================================================================

echo "Step 1.5: Validating data split (checking for data leakage)..."

python test_data_split.py "$FEATURE_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Data split validation failed!"
    exit 1
fi

echo "Data split validation passed!"

echo "Step 1.6: Testing evaluation system..."

python test_emotion_evaluation.py

if [ $? -ne 0 ]; then
    echo "Warning: Evaluation system test had issues (but continuing...)"
else
    echo "Evaluation system test passed!"
fi

# =============================================================================
# Training
# =============================================================================

echo "Step 2: Training emotion regression model (Cnn6)..."

# Test mixup fix first
echo "Testing mixup fix..."
python test_mixup_fix.py

if [ $? -ne 0 ]; then
    echo "Mixup test failed, training without mixup augmentation"
    AUGMENTATION="none"
else
    echo "Mixup test passed, using mixup augmentation"
    AUGMENTATION="mixup"
fi

python pytorch/emotion_main.py train \
    --dataset_path "$FEATURE_FILE" \
    --workspace "$WORKSPACE" \
    --model_type "FeatureEmotionRegression_Cnn6" \
    --pretrained_checkpoint_path "$PRETRAINED_MODEL" \
    --freeze_base \
    --loss_type "mse" \
    --augmentation "$AUGMENTATION" \
    --learning_rate 1e-4 \
    --batch_size 16 \
    --stop_iteration 5000 \
    --cuda

echo "Training completed!"

echo ""
echo "=== Training Notes ==="
echo "- Model uses 70% train / 30% validation split by audio files"
echo "- Validation metrics are computed at both segment-level and audio-level"
echo "- Audio-level metrics (aggregated) are the primary evaluation metrics"
echo "- Training logs show both segment and audio-level performance"
echo "- Look for 'Audio Mean MAE' and 'Audio Mean Pearson' in logs for best indicators"

# =============================================================================
# Evaluation
# =============================================================================

echo "Step 3: Evaluating trained model..."

# Find the latest checkpoint
LATEST_CHECKPOINT=$(find "$WORKSPACE/checkpoints" -name "*.pth" | sort -V | tail -n 1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found for evaluation!"
    exit 1
fi

echo "Using checkpoint: $LATEST_CHECKPOINT"

python pytorch/emotion_main.py inference \
    --model_path "$LATEST_CHECKPOINT" \
    --dataset_path "$FEATURE_FILE" \
    --model_type "FeatureEmotionRegression_Cnn6" \
    --batch_size 32 \
    --cuda

echo "Evaluation completed!"

# =============================================================================
# Usage Information
# =============================================================================

echo ""
echo "=== Training Complete ==="
echo "Model checkpoints saved in: $WORKSPACE/checkpoints"
echo "Training logs saved in: $WORKSPACE/logs"
echo "Training statistics saved in: $WORKSPACE/statistics"
echo ""
echo "To resume training from a checkpoint, use:"
echo "python pytorch/emotion_main.py train --resume_iteration <iteration> [other args...]"
echo ""
echo "To evaluate a specific checkpoint, use:"
echo "python pytorch/emotion_main.py inference --model_path <checkpoint_path> [other args...]" 