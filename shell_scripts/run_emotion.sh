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
    # Try relative path as fallback
    elif [ -f "features/emotion_features/emotion_features.h5" ]; then
        FEATURE_FILE="features/emotion_features/emotion_features.h5"
    # Try absolute path if provided by user (update this if features are elsewhere)
    elif [ -f "/path/to/your/emotion_features.h5" ]; then
        FEATURE_FILE="/path/to/your/emotion_features.h5"
    else
        echo "Error: No existing features found!"
        echo "Searched in the following locations:"
        echo "  - $WORKSPACE/features/emotion_features.h5"
        echo "  - features/emotion_features/emotion_features.h5"
        echo ""
        echo "Solutions:"
        echo "  1. Run without --skip-extraction to extract features"
        echo "  2. Update the FEATURE_FILE path in this script to point to your existing features"
        echo "  3. Copy your features to one of the expected locations"
        exit 1
    fi
fi

# Path to pretrained PANNs model (download from PANNs repository)
PRETRAINED_MODEL="/DATA/pliu/EmotionData/Cnn6_mAP=0.343.pth"  # Cnn6 model
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
    echo "Using existing features at $FEATURE_FILE"
    echo "⚠️  WARNING: If you're using old features (not segmented), you may need to re-extract!"
    echo "   The updated system expects 6 segments per audio file (7278 total samples)"
    echo "   Old format has 1 sample per audio file (1213 total samples)"
else
    echo "Step 1: Extracting features from Emo-Soundscapes dataset..."

    PYTHONPATH=. python3 scripts/extract_features.py \
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

PYTHONPATH=. python3 tests/test_data_split.py "$FEATURE_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Data split validation failed!"
    exit 1
fi

echo "Data split validation passed!"

echo "Step 1.6: Testing evaluation system..."

PYTHONPATH=. python3 tests/test_emotion_evaluation.py

if [ $? -ne 0 ]; then
    echo "Warning: Evaluation system test had issues (but continuing...)"
else
    echo "Evaluation system test passed!"
fi

# =============================================================================
# Training Configuration for 12GB GPU
# =============================================================================

# Updated configuration for epoch-based training
BATCH_SIZE=32        # Optimized batch size for consistent comparison
EPOCHS=100           # More intuitive than iterations
LEARNING_RATE=0.001  # Higher learning rate for better convergence

# Calculate approximate iterations for 100 epochs
# Assuming ~2000 training samples: 2000/48 ≈ 42 iterations/epoch
# 100 epochs ≈ 4200 iterations (will auto-adjust based on actual dataset size)
STOP_ITERATION=20000  # Extended training for better convergence

echo "🚀 Training Configuration:"
echo "  - Batch Size: $BATCH_SIZE (optimized for consistent comparison)"
echo "  - Target Epochs: $EPOCHS"
echo "  - Estimated Iterations: $STOP_ITERATION"
echo "  - Learning Rate: $LEARNING_RATE"
echo ""

# =============================================================================
# Training
# =============================================================================

echo "Step 2: Training emotion regression model (Cnn6 with New Affective System)..."
echo "🎯 Training for $EPOCHS epochs with batch size $BATCH_SIZE"

# Force mixup augmentation for consistent comparison
echo "Using mixup augmentation (forced on for fair comparison)"
AUGMENTATION="mixup"

PYTHONPATH=. python3 scripts/train.py \
    --dataset_path "$FEATURE_FILE" \
    --workspace "$WORKSPACE" \
    --model_type "FeatureEmotionRegression_Cnn6_NewAffective" \
    --pretrained_checkpoint_path "$PRETRAINED_MODEL" \
    --freeze_base \
    --loss_type "mse" \
    --augmentation "$AUGMENTATION" \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --stop_iteration $STOP_ITERATION \
    --cuda

echo "Training completed!"

echo ""
echo "=== Training Notes ==="
echo "- Model: FeatureEmotionRegression_Cnn6_NewAffective"
echo "- Epochs: $EPOCHS (approx)"
echo "- Batch Size: $BATCH_SIZE (optimized for consistent comparison)"
echo "- Architecture: Frozen CNN6 visual system + New separate affective pathways"
echo "- Visual System: CNN6 backbone (frozen, pretrained on AudioSet)"
echo "- Affective System: Separate valence/arousal pathways (512→256→128→1)"
echo "- No Feedback: Pure feedforward processing (compare with LRM feedback model)"
echo "- Model uses 70% train / 30% validation split by audio files"
echo "- Validation metrics are computed at both segment-level and audio-level"
echo "- Audio-level metrics (aggregated) are the primary evaluation metrics"
echo "- Look for 'Audio Mean MAE' and 'Audio Mean Pearson' in logs for best indicators"

# =============================================================================
# Evaluation
# =============================================================================

echo "Step 3: Evaluating trained model with CSV export and visualizations..."

# Find the best model checkpoint (preferred) or latest checkpoint as fallback
BEST_MODEL_PATH=$(find "$WORKSPACE/checkpoints" -name "best_model.pth" | head -n 1)

if [ -n "$BEST_MODEL_PATH" ]; then
    CHECKPOINT_TO_USE="$BEST_MODEL_PATH"
    echo "Using BEST model checkpoint: $CHECKPOINT_TO_USE"
else
    # Fallback to latest checkpoint if best model not found
    LATEST_CHECKPOINT=$(find "$WORKSPACE/checkpoints" -name "*.pth" | sort -V | tail -n 1)
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "No checkpoint found for evaluation!"
        exit 1
    fi
    CHECKPOINT_TO_USE="$LATEST_CHECKPOINT"
    echo "⚠️  Best model not found, using latest checkpoint: $CHECKPOINT_TO_USE"
fi

PYTHONPATH=. python3 scripts/evaluate.py \
    --model_path "$CHECKPOINT_TO_USE" \
    --dataset_path "$FEATURE_FILE" \
    --model_type "FeatureEmotionRegression_Cnn6_NewAffective" \
    --batch_size 32 \
    --cuda

echo "Evaluation completed!"

# Check if predictions were generated
PREDICTIONS_DIR="$WORKSPACE/predictions"

if [ -d "$PREDICTIONS_DIR" ]; then
    echo ""
    echo "=== Generated Files ==="
    echo "Predictions saved in: $PREDICTIONS_DIR"
    echo "- segment_predictions.csv: Segment-level predictions with time information"
    echo "- audio_predictions.csv: Audio-level aggregated predictions"
    echo "- plots/: Visualization plots including:"
    echo "  - audio_scatter_plots.png: True vs predicted scatter plots (audio-level)"
    echo "  - segment_scatter_plots.png: True vs predicted scatter plots (segment-level)"
    echo "  - time_series_sample.png: Sample time-series plots"
    echo "  - individual_timeseries/: Individual time-series for each audio file"
    echo "  - summary_statistics.png: Error distributions and performance analysis"
else
    echo "Warning: Predictions directory not found. Visualizations may not have been generated."
fi

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