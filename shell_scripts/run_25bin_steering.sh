#!/bin/bash

# Script to run the optimal 25-bin steering test
# Based on the scientific findings from STEERING_SIGNALS_FINAL_REPORT.md
# This script runs the 25-bin steering approach with strength optimization

# =============================================================================
# Configuration - MODIFY THESE PATHS TO MATCH YOUR SETUP
# =============================================================================

# Path to your emotion dataset
DATASET_PATH="workspaces/emotion_feedback/features/emotion_features.h5"

# Path to trained model checkpoint (use the best model from training)
MODEL_CHECKPOINT="workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=16/freeze_base=True/best_model.pth"

# Path to 25-bin steering signals
STEERING_SIGNALS_PATH="steering_signals_25bin/steering_signals_25bin.json"

# Output directory for results
RESULTS_DIR="steering_test_results_25bin"

# =============================================================================
# Validation Checks
# =============================================================================

echo "🎯 Running Optimal 25-Bin Steering Test"
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

# Check if 25-bin steering signals exist
if [ ! -f "$STEERING_SIGNALS_PATH" ]; then
    echo "❌ Error: 25-bin steering signals not found: $STEERING_SIGNALS_PATH"
    echo "Please run the 25-bin steering signal generation first:"
    echo "  python scripts/generate_25bin_steering_signals.py"
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
# Run the Optimal 25-Bin Steering Test
# =============================================================================

echo "🔬 Running Optimal 25-Bin Steering Test..."
echo "🎯 Approach: 25-bin categorical steering with strength optimization"
echo "📊 Testing strengths: 0.0, 1.0, 1.5, 2.0, 2.5, 3, 3.5, 4, 4.5, 5.0, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10"
echo ""

# Run the optimal 25-bin steering test
PYTHONPATH=src:. python3 scripts/test_steering.py \
    --dataset "$DATASET_PATH" \
    --model "$MODEL_CHECKPOINT" \
    --steering_signals "$STEERING_SIGNALS_PATH" \
    --output "$RESULTS_DIR/steering_results_25bin.json" \
    --max_samples 100000

if [ $? -ne 0 ]; then
    echo "❌ Error: 25-bin steering test failed!"
    exit 1
fi

echo ""
echo "✅ 25-bin steering test completed successfully!"
echo ""

# =============================================================================
# Display Results
# =============================================================================

echo "📊 25-BIN STEERING PERFORMANCE RESULTS"
echo "======================================"

# Check if results file exists and display table
if [ -f "$RESULTS_DIR/steering_results_25bin.json" ]; then
    echo "Loading results from: $RESULTS_DIR/steering_results_25bin.json"
    echo ""
    
    # Use Python to parse and display the table
    python3 -c "
import json
import sys

try:
    with open('$RESULTS_DIR/steering_results_25bin.json', 'r') as f:
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
    
    best_val_strength = max(results.keys(), key=lambda s: results[s]['val_corr'] - baseline['val_corr'] if s != '0.0' else -999)
    best_val_improvement = results[best_val_strength]['val_corr'] - baseline['val_corr']
    
    print()
    print(f\"🏆 Best valence improvement: +{best_val_improvement:.4f} at strength {best_val_strength}\")
    print(f\"🏆 Best arousal improvement: +{best_aro_improvement:.4f} at strength {best_aro_strength}\")
    
    # Compare with 9-bin expected results
    print()
    print(f\"📈 COMPARISON WITH 9-BIN EXPECTED RESULTS:\")
    print(f\"   9-bin expected arousal improvement: +0.014 at strength 2.0\")
    print(f\"   25-bin actual arousal improvement: +{best_aro_improvement:.4f} at strength {best_aro_strength}\")
    
    if best_aro_improvement > 0.014:
        print(f\"   ✅ BETTER: 25-bin outperforms 9-bin expected results!\")
    elif best_aro_improvement >= 0.010:
        print(f\"   ✅ GOOD: 25-bin achieves significant improvement\")
    else:
        print(f\"   ⚠️  MARGINAL: 25-bin shows small improvement\")
    
except Exception as e:
    print(f\"Error reading results: {e}\")
    print(\"Please check if the test completed successfully.\")
"
else
    echo "❌ Results file not found. Please check if the test completed successfully."
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "🎉 25-Bin Steering Test Completed!"
echo ""
echo "Summary:"
echo "  - Tested 25-bin steering signals with strength optimization"
echo "  - Used production-ready test_steering.py approach"
echo "  - Compared results with 9-bin expected performance"
echo "  - Found optimal strength for maximum improvement"
echo ""
echo "Key Benefits of 25-bin vs 9-bin:"
echo "  - Finer emotion categorization (5x5 vs 3x3 grid)"
echo "  - More precise steering signals for specific emotion regions"
echo "  - Potentially better performance due to finer granularity"
echo ""
echo "Results saved to: $RESULTS_DIR/steering_results_25bin.json" 