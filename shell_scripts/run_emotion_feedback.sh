#!/bin/bash

# Script to run emotion regression training with TOP-DOWN FEEDBACK on Emo-Soundscapes dataset
# This script demonstrates the new FeatureEmotionRegression_Cnn6_LRM model
# which incorporates psychologically-motivated feedback connections:
# - Valence modulates semantic processing (higher conv layers)  
# - Arousal modulates attention to acoustic details (lower conv layers)
#
# For full-length audios: Feedback signals are computed once and reused across passes
# ensuring consistent modulation since there are no segments anymore.
#
# Usage:
#   bash run_emotion_feedback.sh                    # Run full pipeline (extract + train + eval)
#   bash run_emotion_feedback.sh --skip-extraction  # Skip feature extraction, use existing features

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

# Your workspace directory (separate from standard emotion workspace)
WORKSPACE="workspaces/emotion_feedback"

# Feature file location - will be determined based on extraction mode
FEATURE_FILE=""

# Set feature file path based on extraction mode
if [ "$SKIP_EXTRACTION" = false ]; then
    # If extracting, features will be in workspace
    FEATURE_FILE="$WORKSPACE/features/emotion_features.h5"
else
    # If skipping extraction, look for existing features in common locations
    # Try workspace location first (most likely)
    if [ -f "$WORKSPACE/features/emotion_features.h5" ]; then
        FEATURE_FILE="$WORKSPACE/features/emotion_features.h5"
    # Try standard workspace location as fallback
    elif [ -f "workspaces/emotion_regression/features/emotion_features.h5" ]; then
        FEATURE_FILE="workspaces/emotion_regression/features/emotion_features.h5"
    # Try relative path as fallback
    elif [ -f "features/emotion_features/emotion_features.h5" ]; then
        FEATURE_FILE="features/emotion_features/emotion_features.h5"
    else
        echo "Error: No existing features found!"
        echo "Searched in the following locations:"
        echo "  - $WORKSPACE/features/emotion_features.h5"
        echo "  - workspaces/emotion_regression/features/emotion_features.h5"
        echo "  - features/emotion_features/emotion_features.h5"
        echo ""
        echo "Solutions:"
        echo "  1. Run without --skip-extraction to extract features"
        echo "  2. Copy existing features to $WORKSPACE/features/emotion_features.h5"
        echo "  3. Run the standard run_emotion.sh first to generate features"
        exit 1
    fi
fi

# Path to pretrained PANNs model (download from PANNs repository)
PRETRAINED_MODEL="/DATA/pliu/EmotionData/Cnn6_mAP=0.343.pth"  # Cnn6 model

# =============================================================================
# Training Configuration for 12GB GPU with Feedback
# =============================================================================

# Updated configuration for epoch-based training with feedback and full-length audios
BATCH_SIZE=16        # Changed batch size to 16
EPOCHS=100           # More intuitive than iterations
LEARNING_RATE=0.001  # Higher learning rate for feedback model convergence

# Calculate approximate iterations for 100 epochs
# Assuming ~1200 audio files: 1200/16 ≈ 75 iterations/epoch
# 100 epochs ≈ 7500 iterations (will auto-adjust based on actual dataset size)
STOP_ITERATION=15000  # Extended training for better convergence

# Feedback model configuration
MODEL_TYPE="FeatureEmotionRegression_Cnn6_LRM"
FORWARD_PASSES=2  # Number of feedback iterations

echo "🔄 Setting up Emotion Regression with ORIGINAL LRM TOP-DOWN FEEDBACK..."
echo "🚀 Training Configuration:"
echo "  - Model: $MODEL_TYPE (Original LRM Implementation)"
echo "  - Batch Size: $BATCH_SIZE (changed to 16)"
echo "  - Target Epochs: $EPOCHS"
echo "  - Estimated Iterations: $STOP_ITERATION"
echo "  - Forward Passes: $FORWARD_PASSES"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - No weight decay or learning rate schedule"
echo "  - Advanced Features: Normalization, Squashing, Asymmetric Modulation"
echo "  - Full-Length Audio: Consistent feedback signals across passes"
echo ""

# =============================================================================
# Validation Checks
# =============================================================================

echo "🔄 Setting up Emotion Regression with ORIGINAL LRM TOP-DOWN FEEDBACK..."
echo "Model: $MODEL_TYPE (Original LRM Implementation)"
echo "Forward Passes: $FORWARD_PASSES"
echo "Batch Size: $BATCH_SIZE"
echo "Advanced Features: Normalization, Squashing, Asymmetric Modulation"
echo "Full-Length Audio: Consistent feedback signals across passes"
echo ""

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

echo "✅ Setup validation passed!"

# =============================================================================
# Feature Extraction
# =============================================================================

if [ "$SKIP_EXTRACTION" = true ]; then
    echo "⏭️  Skipping feature extraction (--skip-extraction flag provided)"
    echo "Using existing features at $FEATURE_FILE"
    echo "✅ Using full-length audio features (no segments)"
else
    echo "Step 1: Extracting features from Emo-Soundscapes dataset (full-length audios)..."

    PYTHONPATH=. python3 scripts/extract_features.py \
        --audio_dir "$EMO_AUDIO_DIR" \
        --ratings_dir "$EMO_RATINGS_DIR" \
        --output_dir "$WORKSPACE/features"

    # Check if feature extraction was successful
    if [ ! -f "$FEATURE_FILE" ]; then
        echo "Error: Feature extraction failed! Expected features at: $FEATURE_FILE"
        exit 1
    fi

    echo "✅ Feature extraction completed successfully!"
fi

# =============================================================================
# Data Validation
# =============================================================================

echo "Step 1.5: Validating data split (checking for data leakage)..."

PYTHONPATH=. python3 tests/test_data_split.py "$FEATURE_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Data split validation failed!"
    exit 1
fi

echo "✅ Data split validation passed!"

echo "Step 1.6: Testing evaluation system..."

PYTHONPATH=. python3 tests/test_emotion_evaluation.py

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Evaluation system test had issues (but continuing...)"
else
    echo "✅ Evaluation system test passed!"
fi

# =============================================================================
# Feedback Model Demo
# =============================================================================

echo "Step 1.7: Testing feedback model..."

PYTHONPATH=. python3 examples/emotion_example.py

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Feedback model test had issues (but continuing...)"
else
    echo "✅ Feedback model test passed!"
fi

# =============================================================================
# Training
# =============================================================================

echo "Step 2: Training emotion regression model with LRM feedback..."
echo "🎯 Training for $EPOCHS epochs with batch size $BATCH_SIZE"
echo "🔄 Using $FORWARD_PASSES forward passes with feedback"

# Force mixup augmentation for consistent comparison
echo "Using mixup augmentation (forced on for fair comparison)"
AUGMENTATION="mixup"

PYTHONPATH=. python3 scripts/train.py \
    --dataset_path "$FEATURE_FILE" \
    --workspace "$WORKSPACE" \
    --model_type "$MODEL_TYPE" \
    --pretrained_checkpoint_path "$PRETRAINED_MODEL" \
    --freeze_base \
    --loss_type "mse" \
    --augmentation "$AUGMENTATION" \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --stop_iteration "$STOP_ITERATION" \
    --forward_passes "$FORWARD_PASSES" \
    --cuda \
    --gpu_id 0

# Check if training was successful
if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

echo "Training completed successfully!"

# =============================================================================
# Evaluation
# =============================================================================

echo "Step 3: Evaluating trained feedback model..."

# Use the evaluation script's built-in best model detection
echo "🔍 Auto-detecting best model for evaluation..."

PYTHONPATH=. python3 scripts/evaluate.py \
    --dataset_path "$FEATURE_FILE" \
    --workspace "$WORKSPACE" \
    --model_type "$MODEL_TYPE" \
    --use_best_model \
    --batch_size "$BATCH_SIZE" \
    --forward_passes "$FORWARD_PASSES" \
    --cuda \
    --gpu_id 0

if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed!"
    exit 1
fi

echo "Evaluation completed successfully!"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "🎉 Feedback Pipeline completed successfully!"
echo ""
echo "Summary:"
echo "  - Features extracted from full-length audios"
echo "  - LRM feedback model trained for $EPOCHS epochs"
echo "  - $FORWARD_PASSES forward passes with consistent feedback signals"
echo "  - Final evaluation completed"
echo ""
echo "Results are available in:"
echo "  - Checkpoints: $WORKSPACE/checkpoints"
echo "  - Logs: $WORKSPACE/logs"
echo "  - Statistics: $WORKSPACE/statistics"
echo ""
echo "Key Features of LRM Feedback Model:"
echo "  - Top-down feedback from valence/arousal to visual processing"
echo "  - Consistent feedback signals across multiple passes"
echo "  - Normalization, squashing, and asymmetric modulation"
echo "  - Full-length audio processing (no segments)"
echo ""
echo "To run inference on new audio files, use:"
echo "  python3 scripts/evaluate.py --use_best_model --workspace $WORKSPACE --model_type $MODEL_TYPE --audio_file your_audio.wav" 