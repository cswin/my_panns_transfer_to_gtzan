#!/bin/bash

# Script to test multiple passes for emotion_feedback model
# Shows performance as a function of number of feed-forward passes
#
# Usage:
#   bash test_multiple_passes.sh                    # Test with default settings
#   bash test_multiple_passes.sh --max-passes 5     # Test up to 5 passes
#   bash test_multiple_passes.sh --model-path /path/to/model.pth  # Use specific model

# =============================================================================
# Configuration
# =============================================================================

# Parse command line arguments
MAX_PASSES=6
MODEL_PATH=""
WORKSPACE="workspaces/emotion_feedback"
DATASET_PATH="workspaces/emotion_feedback/features/emotion_features.h5"
BATCH_SIZE=16
NUM_SAMPLES=100  # Number of samples to test for each pass count

for arg in "$@"; do
    case $arg in
        --max-passes)
            MAX_PASSES="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --workspace)
            WORKSPACE="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --help)
            echo "Usage: bash test_multiple_passes.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --max-passes N       Maximum number of passes to test (default: 6)"
            echo "  --model-path PATH    Path to specific model checkpoint"
            echo "  --workspace PATH     Workspace directory (default: workspaces/emotion_feedback)"
            echo "  --dataset-path PATH  Path to emotion features HDF5 file"
            echo "  --batch-size N       Batch size for evaluation (default: 16)"
            echo "  --num-samples N      Number of samples to test per pass (default: 100, use 0 for all samples)"
            echo "  --help               Show this help message"
            exit 0
            ;;
    esac
done

# =============================================================================
# Auto-detect paths if not provided
# =============================================================================

# Auto-detect dataset path if not provided
if [ -z "$DATASET_PATH" ]; then
    if [ -f "$WORKSPACE/features/emotion_features.h5" ]; then
        DATASET_PATH="$WORKSPACE/features/emotion_features.h5"
    elif [ -f "workspaces/emotion_regression/features/emotion_features.h5" ]; then
        DATASET_PATH="workspaces/emotion_regression/features/emotion_features.h5"
    elif [ -f "features/emotion_features/emotion_features.h5" ]; then
        DATASET_PATH="features/emotion_features/emotion_features.h5"
    else
        echo "Error: Could not auto-detect dataset path!"
        echo "Please provide --dataset-path argument"
        exit 1
    fi
fi

# Auto-detect model path if not provided
if [ -z "$MODEL_PATH" ]; then
    # Look for best model in workspace with specific path structure
    if [ -f "$WORKSPACE/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=16/freeze_base=True/best_model.pth" ]; then
        MODEL_PATH="$WORKSPACE/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=16/freeze_base=True/best_model.pth"
    elif [ -f "$WORKSPACE/checkpoints/best_model.pth" ]; then
        MODEL_PATH="$WORKSPACE/checkpoints/best_model.pth"
    elif [ -f "$WORKSPACE/checkpoints/model_*.pth" ]; then
        # Get the most recent model file
        MODEL_PATH=$(ls -t "$WORKSPACE/checkpoints/model_*.pth" | head -1)
    else
        echo "Error: Could not auto-detect model path!"
        echo "Please provide --model-path argument"
        exit 1
    fi
fi

# =============================================================================
# Validation
# =============================================================================

echo "🧪 Testing Multiple Passes for Emotion Feedback Model"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  - Max Passes: $MAX_PASSES"
echo "  - Model Path: $MODEL_PATH"
echo "  - Dataset Path: $DATASET_PATH"
echo "  - Workspace: $WORKSPACE"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Samples per Test: $NUM_SAMPLES"
echo ""

# Check if files exist
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

# =============================================================================
# Create Python script for testing multiple passes
# =============================================================================

echo "📝 Creating test script..."

# Make the script executable
chmod +x scripts/test_multiple_passes.py

# =============================================================================
# Run the test
# =============================================================================

echo "🚀 Running multiple passes test..."

# Create results directory
RESULTS_DIR="$WORKSPACE/multiple_passes_test"
mkdir -p "$RESULTS_DIR"

# Run the test
PYTHONPATH=. python3 scripts/test_multiple_passes.py \
    --model-path "$MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --max-passes "$MAX_PASSES" \
    --batch-size "$BATCH_SIZE" \
    --num-samples "$NUM_SAMPLES" \
    --output-dir "$RESULTS_DIR"

# Check if test was successful
if [ $? -ne 0 ]; then
    echo "Error: Multiple passes test failed!"
    exit 1
fi

# =============================================================================
# Display results
# =============================================================================

echo ""
echo "🎉 Multiple passes testing completed successfully!"
echo ""
echo "📁 Results saved in: $RESULTS_DIR"
echo ""
echo "Files generated:"
echo "  - multiple_passes_results.csv: Detailed results table"
echo "  - multiple_passes_results.json: Results in JSON format"
echo "  - multiple_passes_performance.png: Comprehensive performance plots"
echo "  - multiple_passes_summary.png: Summary plot with performance vs cost"
echo ""

# Display summary if results file exists
if [ -f "$RESULTS_DIR/multiple_passes_results.csv" ]; then
    echo "📊 Quick Summary:"
    echo "================="
    
    # Use Python to display a nice summary
    python3 -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$RESULTS_DIR/multiple_passes_results.csv')
    
    print('Passes | Mean MAE | Mean Pearson | Time (ms)')
    print('-------|----------|--------------|----------')
    
    for _, row in df.iterrows():
        print(f'{row[\"num_passes\"]:6.0f} | {row[\"mean_mae\"]:8.4f} | {row[\"mean_pearson\"]:12.4f} | {row[\"avg_processing_time_ms\"]:8.1f}')
    
    # Find best performance
    best_mae_idx = df['mean_mae'].idxmin()
    best_pearson_idx = df['mean_pearson'].idxmin()
    
    print('')
    print(f'🎯 Best MAE: {df.loc[best_mae_idx, \"mean_mae\"]:.4f} at {df.loc[best_mae_idx, \"num_passes\"]} passes')
    print(f'🎯 Best Pearson: {df.loc[best_pearson_idx, \"mean_pearson\"]:.4f} at {df.loc[best_pearson_idx, \"num_passes\"]} passes')
    
except Exception as e:
    print(f'Error reading results: {e}')
    sys.exit(1)
"
fi

echo ""
echo "🔍 Analysis complete! Check the plots in $RESULTS_DIR for visual analysis."
echo ""
echo "💡 Key insights to look for:"
echo "  - Optimal number of passes for best performance"
echo "  - Performance saturation point"
echo "  - Computational cost vs performance trade-off"
echo "  - Whether feedback actually improves performance"
echo ""
echo "📈 To view plots:"
echo "  open $RESULTS_DIR/multiple_passes_performance.png"
echo "  open $RESULTS_DIR/multiple_passes_summary.png"
