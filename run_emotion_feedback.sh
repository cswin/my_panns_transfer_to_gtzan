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

# Feedback model configuration
MODEL_TYPE="FeatureEmotionRegression_Cnn6_LRM"
FORWARD_PASSES=2  # Number of feedback iterations
BATCH_SIZE=16     # Smaller batch size for feedback model (uses more memory)

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
echo "Model: $MODEL_TYPE"
echo "Forward Passes: $FORWARD_PASSES"
echo "Batch Size: $BATCH_SIZE"
echo ""

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
    --model_type "$MODEL_TYPE" \
    --pretrained_checkpoint_path "$PRETRAINED_MODEL" \
    --freeze_base \
    --loss_type "mse" \
    --augmentation "$AUGMENTATION" \
    --learning_rate 1e-4 \
    --batch_size "$BATCH_SIZE" \
    --forward_passes "$FORWARD_PASSES" \
    --stop_iteration 5000 \
    --cuda

echo "✅ Training completed!"

echo ""
echo "🔄 === Feedback Training Notes ==="
echo "- Model: $MODEL_TYPE with $FORWARD_PASSES forward passes"
echo "- Each forward pass uses feedback from previous emotion predictions"
echo "- Valence predictions modulate conv3 and conv4 features"
echo "- Arousal predictions modulate conv2 features"  
echo "- Training uses 70% train / 30% validation split by audio files"
echo "- Look for 'Audio Mean MAE' and 'Audio Mean Pearson' in logs"

# =============================================================================
# Evaluation with Feedback
# =============================================================================

echo "🎯 Step 3: Evaluating feedback model with CSV export and visualizations..."

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