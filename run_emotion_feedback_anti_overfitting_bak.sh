#!/bin/bash

# Improved LRM training script with gradient stability measures
# This script addresses training instability from feedback-induced gradient complexity

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

WORKSPACE_PATH="workspaces/emotion_feedback_stable"
PRETRAINED_MODEL_PATH="pretrained_model/Cnn6_mAP=0.343.pth"

# Paths for feature extraction (same as run_emotion.sh)
EMO_AUDIO_DIR="/DATA/pliu/EmotionData/Emo-Soundscapes/audio_flat"
EMO_RATINGS_DIR="/DATA/pliu/EmotionData/Emo-Soundscapes/Emo-Soundscapes-Ratings"

# Feature file location - will be determined based on extraction mode
DATASET_PATH=""

# Set feature file path based on extraction mode
if [ "$SKIP_EXTRACTION" = false ]; then
    # If extracting, features will be in workspace
    DATASET_PATH="$WORKSPACE_PATH/features/emotion_features.h5"
else
    # If skipping extraction, look for existing features in common locations
    # Try workspace location first (most likely)
    if [ -f "$WORKSPACE_PATH/features/emotion_features.h5" ]; then
        DATASET_PATH="$WORKSPACE_PATH/features/emotion_features.h5"
    # Try standard workspace location as fallback
    elif [ -f "workspaces/emotion_regression/features/emotion_features.h5" ]; then
        DATASET_PATH="workspaces/emotion_regression/features/emotion_features.h5"
    # Try feedback workspace location
    elif [ -f "workspaces/emotion_feedback/features/emotion_features.h5" ]; then
        DATASET_PATH="workspaces/emotion_feedback/features/emotion_features.h5"
    # Try relative path as fallback
    elif [ -f "features/emotion_features/emotion_features.h5" ]; then
        DATASET_PATH="features/emotion_features/emotion_features.h5"
    else
        echo "Error: No existing features found!"
        echo "Searched in the following locations:"
        echo "  - $WORKSPACE_PATH/features/emotion_features.h5"
        echo "  - workspaces/emotion_regression/features/emotion_features.h5"
        echo "  - workspaces/emotion_feedback/features/emotion_features.h5"
        echo "  - features/emotion_features/emotion_features.h5"
        echo ""
        echo "Solutions:"
        echo "  1. Run without --skip-extraction to extract features"
        echo "  2. Copy existing features to $WORKSPACE_PATH/features/emotion_features.h5"
        echo "  3. Run the standard run_emotion.sh first to generate features"
        exit 1
    fi
fi

# Create workspace directory
mkdir -p "$WORKSPACE_PATH"
mkdir -p "$WORKSPACE_PATH/checkpoints"
mkdir -p "$WORKSPACE_PATH/logs"
mkdir -p "$WORKSPACE_PATH/statistics"

# =============================================================================
# Feature Extraction
# =============================================================================

if [ "$SKIP_EXTRACTION" = true ]; then
    echo "Skipping feature extraction (--skip-extraction flag provided)"
    echo "Using existing features at $DATASET_PATH"
else
    echo "Step 1: Extracting features from Emo-Soundscapes dataset..."

    python extract_emotion_features.py \
        --audio_dir "$EMO_AUDIO_DIR" \
        --ratings_dir "$EMO_RATINGS_DIR" \
        --output_dir "$WORKSPACE_PATH/features"

    # Check if feature extraction was successful
    if [ ! -f "$DATASET_PATH" ]; then
        echo "Error: Feature extraction failed! Expected features at: $DATASET_PATH"
        exit 1
    fi

    echo "Feature extraction completed successfully!"
fi

echo "Training LRM model with normalized configuration..."
echo "Dataset: $DATASET_PATH"
echo "Workspace: $WORKSPACE_PATH"

# Training with normalized configuration for fair comparison:
python pytorch/emotion_main.py train \
    --dataset_path="$DATASET_PATH" \
    --workspace="$WORKSPACE_PATH" \
    --model_type="FeatureEmotionRegression_Cnn6_LRM" \
    --pretrained_checkpoint_path="$PRETRAINED_MODEL_PATH" \
    --freeze_base \
    --loss_type="mse" \
    --augmentation="mixup" \
    --learning_rate=0.001 \
    --batch_size=32 \
    --resume_iteration=0 \
    --stop_iteration=20000 \
    --forward_passes=2 \
    --cuda

echo "✅ Training completed!"

# =============================================================================
# Evaluation with Prediction Saving (Missing from original anti-overfitting script)
# =============================================================================

echo "🎯 Step 2: Evaluating model with CSV export and visualizations..."

# Find the best model checkpoint (preferred) or latest checkpoint as fallback
BEST_MODEL_PATH=$(find "$WORKSPACE_PATH/checkpoints" -name "best_model.pth" | head -n 1)

if [ -n "$BEST_MODEL_PATH" ]; then
    CHECKPOINT_TO_USE="$BEST_MODEL_PATH"
    echo "Using BEST model checkpoint: $CHECKPOINT_TO_USE"
else
    # Fallback to latest checkpoint if best model not found
    LATEST_CHECKPOINT=$(find "$WORKSPACE_PATH/checkpoints" -name "*.pth" | sort -V | tail -n 1)
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "No checkpoint found for evaluation!"
        exit 1
    fi
    CHECKPOINT_TO_USE="$LATEST_CHECKPOINT"
    echo "⚠️  Best model not found, using latest checkpoint: $CHECKPOINT_TO_USE"
fi

python pytorch/emotion_main.py inference \
    --model_path "$CHECKPOINT_TO_USE" \
    --dataset_path "$DATASET_PATH" \
    --model_type "FeatureEmotionRegression_Cnn6_LRM" \
    --batch_size 32 \
    --forward_passes 2 \
    --cuda

echo "✅ Evaluation completed!"

# Check if predictions were generated
PREDICTIONS_DIR="$WORKSPACE_PATH/predictions"

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

echo ""
echo "🎉 === LRM Training Complete ==="
echo "Workspace: $WORKSPACE_PATH"
echo "Model: FeatureEmotionRegression_Cnn6_LRM"
echo ""
echo "📁 Generated Files:"
echo "- Model checkpoints: $WORKSPACE_PATH/checkpoints"
echo "- Training logs: $WORKSPACE_PATH/logs/"
echo "- Training statistics: $WORKSPACE_PATH/statistics"
echo "- Predictions: $WORKSPACE_PATH/predictions"
echo ""
echo "🔧 To resume training from a checkpoint:"
echo "python pytorch/emotion_main.py train --resume_iteration <iteration> \\"
echo "    --model_type FeatureEmotionRegression_Cnn6_LRM --forward_passes 2 [other args...]" 