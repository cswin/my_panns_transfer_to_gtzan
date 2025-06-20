#!/bin/bash

# Script to run emotion regression training with TOP-DOWN FEEDBACK on Emo-Soundscapes dataset
# This script demonstrates the new FeatureEmotionRegression_Cnn6_LRM model
# which incorporates psychologically-motivated feedback connections:
# - Valence modulates semantic processing (higher conv layers)  
# - Arousal modulates attention to acoustic details (lower conv layers)
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
PRETRAINED_MODEL="pretrained_model/Cnn6_mAP=0.343.pth"  # Cnn6 model

# =============================================================================
# Training Configuration for 12GB GPU with Feedback
# =============================================================================

# Updated configuration for epoch-based training with feedback
BATCH_SIZE=32        # Optimized batch size for stable feedback training
EPOCHS=100           # More intuitive than iterations
LEARNING_RATE=0.001  # Higher learning rate for feedback model convergence

# Calculate approximate iterations for 100 epochs
# Assuming ~2000 training samples: 2000/48 ≈ 42 iterations/epoch
# 100 epochs ≈ 4200 iterations (will auto-adjust based on actual dataset size)
STOP_ITERATION=20000  # Extended training for better convergence

# Feedback model configuration
MODEL_TYPE="FeatureEmotionRegression_Cnn6_LRM"
FORWARD_PASSES=2  # Number of feedback iterations

echo "🔄 Setting up Emotion Regression with ORIGINAL LRM TOP-DOWN FEEDBACK..."
echo "🚀 Training Configuration:"
echo "  - Model: $MODEL_TYPE (Original LRM Implementation)"
echo "  - Batch Size: $BATCH_SIZE (optimized for stable feedback training)"
echo "  - Target Epochs: $EPOCHS"
echo "  - Estimated Iterations: $STOP_ITERATION"
echo "  - Forward Passes: $FORWARD_PASSES"
echo "  - Learning Rate: $LEARNING_RATE"
echo "  - Advanced Features: Normalization, Squashing, Asymmetric Modulation"
echo ""

# =============================================================================
# Validation Checks
# =============================================================================

echo "🔄 Setting up Emotion Regression with ORIGINAL LRM TOP-DOWN FEEDBACK..."
echo "Model: $MODEL_TYPE (Original LRM Implementation)"
echo "Forward Passes: $FORWARD_PASSES"
echo "Batch Size: $BATCH_SIZE"
echo "Advanced Features: Normalization, Squashing, Asymmetric Modulation"
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

    echo "✅ Feature extraction completed successfully!"
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

echo "✅ Data split validation passed!"

echo "Step 1.6: Testing evaluation system..."

python test_emotion_evaluation.py

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Evaluation system test had issues (but continuing...)"
else
    echo "✅ Evaluation system test passed!"
fi

# =============================================================================
# Feedback Model Demo
# =============================================================================

echo "Step 1.7: Testing feedback model..."

python example_cnn6_feedback_emotion.py

if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Feedback model test had issues (but continuing...)"
else
    echo "✅ Feedback model test passed!"
fi

# =============================================================================
# Training with Feedback
# =============================================================================

echo ""
echo "🚀 Step 2: Training emotion regression model with TOP-DOWN FEEDBACK..."
echo "🎯 Training for $EPOCHS epochs with batch size $BATCH_SIZE"
echo "Model: $MODEL_TYPE"
echo "Forward Passes: $FORWARD_PASSES"
echo ""

# Force mixup augmentation for consistent comparison
echo "Using mixup augmentation (forced on for fair comparison)"
AUGMENTATION="mixup"

python pytorch/emotion_main.py train \
    --dataset_path "$FEATURE_FILE" \
    --workspace "$WORKSPACE" \
    --model_type "$MODEL_TYPE" \
    --pretrained_checkpoint_path "$PRETRAINED_MODEL" \
    --freeze_base \
    --loss_type "mse" \
    --augmentation "$AUGMENTATION" \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --stop_iteration $STOP_ITERATION \
    --forward_passes $FORWARD_PASSES \
    --cuda

echo "✅ Training with feedback completed!"

echo ""
echo "=== Training Notes ==="
echo "- Model: $MODEL_TYPE (LRM with TOP-DOWN FEEDBACK)"
echo "- Epochs: $EPOCHS (approx)"
echo "- Batch Size: $BATCH_SIZE (optimized for stable feedback training)"
echo "- Forward Passes: $FORWARD_PASSES"
echo "- Architecture: CNN6 + LRM feedback connections"
echo "- Feedback: Valence→semantic processing, Arousal→acoustic details"

# =============================================================================
# Evaluation with Feedback
# =============================================================================

echo "🎯 Step 3: Evaluating feedback model with CSV export and visualizations..."

# Find the best model checkpoint (preferred for LRM models) or latest checkpoint as fallback
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

python pytorch/emotion_main.py inference \
    --model_path "$CHECKPOINT_TO_USE" \
    --dataset_path "$FEATURE_FILE" \
    --model_type "$MODEL_TYPE" \
    --batch_size 32 \
    --forward_passes "$FORWARD_PASSES" \
    --cuda

echo "✅ Evaluation completed!"

# Check if predictions were generated
PREDICTIONS_DIR="$WORKSPACE/predictions"

if [ -d "$PREDICTIONS_DIR" ]; then
    echo ""
    echo "📊 === Generated Files ==="
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
    echo "⚠️  Warning: Predictions directory not found. Visualizations may not have been generated."
fi

# =============================================================================
# Comparison with Standard Model (if available)
# =============================================================================

echo ""
echo "💡 === Feedback Model vs Standard Model ==="
echo ""

STANDARD_WORKSPACE="workspaces/emotion_regression"
STANDARD_PREDICTIONS="$STANDARD_WORKSPACE/predictions"

if [ -d "$STANDARD_PREDICTIONS" ]; then
    echo "📈 Performance Comparison Available:"
    echo "- Feedback Model Results: $PREDICTIONS_DIR"
    echo "- Standard Model Results: $STANDARD_PREDICTIONS"
    echo ""
    echo "To compare results, examine the CSV files in both directories"
    echo "Key metrics to compare:"
    echo "  - Audio Mean MAE (lower is better)"
    echo "  - Audio Mean Pearson (higher is better)"
    echo "  - Valence/Arousal individual performance"
else
    echo "🔍 To compare with standard model:"
    echo "  1. Run: bash run_emotion.sh --skip-extraction"
    echo "  2. Compare results in both workspace directories"
fi

# =============================================================================
# Usage Information
# =============================================================================

echo ""
echo "🎉 === Feedback Training Complete ==="
echo "Workspace: $WORKSPACE"
echo "Model: $MODEL_TYPE"
echo "Forward Passes: $FORWARD_PASSES"
echo ""
echo "📁 Generated Files:"
echo "- Model checkpoints: $WORKSPACE/checkpoints"
echo "- Training logs: $WORKSPACE/logs"
echo "- Training statistics: $WORKSPACE/statistics"
echo "- Predictions: $WORKSPACE/predictions"
echo ""
echo "🔧 To resume training from a checkpoint:"
echo "python pytorch/emotion_main.py train --resume_iteration <iteration> \\"
echo "    --model_type $MODEL_TYPE --forward_passes $FORWARD_PASSES [other args...]"
echo ""
echo "📊 To evaluate a specific checkpoint:"
echo "python pytorch/emotion_main.py inference --model_path <checkpoint_path> \\"
echo "    --model_type $MODEL_TYPE --forward_passes $FORWARD_PASSES [other args...]"
echo ""
echo "🧪 To test different feedback configurations:"
echo "  - Modify FORWARD_PASSES in this script (try 1, 2, 3, or 4)"
echo "  - Adjust BATCH_SIZE if memory issues occur"
echo "  - Compare results with standard model using run_emotion.sh" 