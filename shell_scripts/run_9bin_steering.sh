#!/bin/bash

# Script to run the optimal 9-bin steering test
# Based on the scientific findings from STEERING_SIGNALS_FINAL_REPORT.md
# This script runs the valence-conv4-only steering approach that achieved +0.014 arousal improvement

# =============================================================================
# Configuration - MODIFY THESE PATHS TO MATCH YOUR SETUP
# =============================================================================

# Path to your emotion dataset
DATASET_PATH="workspaces/emotion_feedback/features/emotion_features.h5"

# Path to trained model checkpoint (use the best model from training)
MODEL_CHECKPOINT="workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=16/freeze_base=True/best_model.pth"

# Path to 9-bin steering signals
STEERING_SIGNALS_PATH="./steering_signal_pairs_9bin/steering_signal_pairs_9bin.json"

# Output directory for results
RESULTS_DIR="steering_test_results"

# =============================================================================
# Validation Checks
# =============================================================================

echo "🎯 Running Optimal 9-Bin Steering Test"
echo "======================================"
echo ""

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset not found: $DATASET_PATH"
    echo "Please update DATASET_PATH in the script to point to your emotion_features.h5 file"
    exit 1
fi

# Check if model checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "❌ Error: Model checkpoint not found: $MODEL_CHECKPOINT"
    echo "Please update MODEL_CHECKPOINT in the script to point to your trained model"
    echo "You can find checkpoints in: workspaces/emotion_feedback/checkpoints/"
    exit 1
fi

# Check if steering signals exist
if [ ! -f "$STEERING_SIGNALS_PATH" ]; then
    echo "❌ Error: 9-bin steering signals not found: $STEERING_SIGNALS_PATH"
    echo "Please run the steering signal generation first:"
    echo "  python scripts/generate_9bin_steering_pairs.py"
    exit 1
fi

echo "✅ All required files found!"
echo ""

# =============================================================================
# Create Results Directory
# =============================================================================

if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
    echo "📁 Created results directory: $RESULTS_DIR"
fi

# =============================================================================
# Run the Optimal Steering Test
# =============================================================================

echo "🔬 Running Optimal 9-Bin Steering Test..."
echo "🎯 Approach: Valence-conv4-only steering"
echo "📊 Testing strengths: 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 5.0, 10.0, 20.0, 50.0"
echo ""

# Run the optimal steering test
PYTHONPATH=src:. python3 scripts/test_steering.py

if [ $? -ne 0 ]; then
    echo "❌ Error: 9-bin steering test failed!"
    exit 1
fi

echo ""
echo "✅ 9-bin steering test completed successfully!"
echo ""





echo "✅ 9-bin steering test completed!"
echo ""

# Display performance table
echo "📊 STEERING PERFORMANCE RESULTS"
echo "================================"

# Check if results file exists and display table
if [ -f "$RESULTS_DIR/steering_results.json" ]; then
    echo "Loading results from: $RESULTS_DIR/steering_results.json"
    echo ""
    
    # Use Python to parse and display the table
    python3 -c "
import json
import sys

try:
    with open('$RESULTS_DIR/steering_results.json', 'r') as f:
        results = json.load(f)
    
    # Get baseline
    baseline = results['0.0']
    
    print(f\"{'Strength':>8} {'Val r':>8} {'Aro r':>8} {'Val Δr':>8} {'Aro Δr':>8} {'Coverage':>8}\")
    print('-' * 60)
    
    for strength in sorted(results.keys(), key=float):
        res = results[strength]
        val_delta = res['val_corr'] - baseline['val_corr']
        aro_delta = res['aro_corr'] - baseline['aro_corr']
        
        print(f\"{float(strength):>8.1f} {res['val_corr']:>8.3f} {res['aro_corr']:>8.3f} \"
              f\"{val_delta:>+8.3f} {aro_delta:>+8.3f} {res['coverage']:>7.1f}%\")
    
    # Find best improvements
    best_aro_strength = max(results.keys(), key=lambda s: results[s]['aro_corr'] - baseline['aro_corr'] if s != '0.0' else -999)
    best_aro_improvement = results[best_aro_strength]['aro_corr'] - baseline['aro_corr']
    
    print()
    print(f\"🏆 Best arousal improvement: +{best_aro_improvement:.4f} at strength {best_aro_strength}\")
    
except Exception as e:
    print(f\"Error reading results: {e}\")
    print(\"Please check if the test completed successfully.\")
"
else
    echo "❌ Results file not found. Please check if the test completed successfully."
fi 