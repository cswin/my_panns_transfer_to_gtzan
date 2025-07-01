#!/bin/bash

# Script to demonstrate external steering signals for emotion feedback testing
# This script shows how to:
# 1. Generate steering signals by categorizing audio data into 9 emotion bins
# 2. Test the steering signals with the trained emotion model
# 3. Visualize the effects of steering on emotion predictions

# =============================================================================
# Configuration - MODIFY THESE PATHS TO MATCH YOUR SETUP
# =============================================================================

# Path to your emotion dataset
DATASET_PATH="workspaces/emotion_feedback/features/emotion_features.h5"

# Path to trained model checkpoint (use the best model from training)
MODEL_CHECKPOINT="workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth"

# Output directories
STEERING_DIR="steering_signals"
TEST_RESULTS_DIR="steering_test_results"

# =============================================================================
# Validation Checks
# =============================================================================

echo "🔄 Setting up External Steering Signals Pipeline..."
echo ""

# Check if dataset exists
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset not found: $DATASET_PATH"
    echo "Please update DATASET_PATH in the script to point to your emotion_features.h5 file"
    exit 1
fi

# Check if model checkpoint exists
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "Error: Model checkpoint not found: $MODEL_CHECKPOINT"
    echo "Please update MODEL_CHECKPOINT in the script to point to your trained model"
    echo "You can find checkpoints in: workspaces/emotion_feedback/checkpoints/"
    exit 1
fi

echo "✅ Dataset and model checkpoint found!"
echo ""

# =============================================================================
# Step 1: Generate Steering Signals
# =============================================================================

echo "Step 1: Generating External Steering Signals"
echo "🎯 Categorizing audio data into 9 emotion bins (3x3 valence/arousal grid)"
echo ""

# Check if steering signals already exist
if [ -d "$STEERING_DIR" ]; then
    echo "⚠️  Steering signals directory already exists: $STEERING_DIR"
    read -p "Do you want to regenerate steering signals? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Regenerating steering signals..."
        rm -rf "$STEERING_DIR"
    else
        echo "Using existing steering signals..."
        SKIP_GENERATION=true
    fi
fi

if [ "$SKIP_GENERATION" != "true" ]; then
    echo "🔄 Running steering signal generation..."
    
    PYTHONPATH=. python3 scripts/generate_steering_signals.py \
        --dataset_path "$DATASET_PATH" \
        --model_checkpoint "$MODEL_CHECKPOINT" \
        --output_dir "$STEERING_DIR" \
        --categorization_method "quantile" \
        --batch_size 32 \
        --cuda \
        --gpu_id 0
    
    if [ $? -ne 0 ]; then
        echo "Error: Steering signal generation failed!"
        exit 1
    fi
    
    echo "✅ Steering signals generated successfully!"
    echo "📁 Output directory: $STEERING_DIR"
    echo "📊 Generated visualizations: emotion_categories.png, category_distribution.png"
    echo ""
fi

# =============================================================================
# Step 2: Test Steering Signals
# =============================================================================

echo "Step 2: Testing Steering Signals with Emotion Model"
echo "🧪 Applying steering signals to test samples and measuring effects"
echo ""

echo "🔄 Running steering signal testing..."
PYTHONPATH=src:. python3 scripts/experimental/test_steering_signals.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --steering_dir "$STEERING_DIR" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$TEST_RESULTS_DIR" \
    --num_samples 20 \
    --batch_size 8 \
    --cuda \
    --gpu_id 0 \
    --steering_method external

if [ $? -ne 0 ]; then
    echo "Error: Steering signal testing failed!"
    exit 1
fi

echo "✅ Steering signal testing completed!"
echo "📁 Results directory: $TEST_RESULTS_DIR"
echo "📊 Generated visualizations: valence_changes.png, arousal_changes.png, steering_effects_2d.png"
echo ""

# =============================================================================
# Step 3: Display Results Summary
# =============================================================================

echo "Step 3: Results Summary"
echo "📈 Analyzing steering signal effects"
echo ""

if [ -f "$TEST_RESULTS_DIR/steering_test_summary.txt" ]; then
    echo "📋 Steering Test Summary:"
    echo "=========================="
    cat "$TEST_RESULTS_DIR/steering_test_summary.txt"
    echo ""
else
    echo "⚠️  Summary file not found. Check the test results directory for details."
fi

# =============================================================================
# Summary
# =============================================================================

echo "🎉 External Steering Signals Pipeline Completed Successfully!"
echo ""
echo "Summary:"
echo "  - Generated 9 steering signal categories based on valence/arousal bins"
echo "  - Tested steering effects on 20 audio samples"
echo "  - Created visualizations of steering signal impacts"
echo ""
echo "Key Files Generated:"
echo "  📁 $STEERING_DIR/"
echo "    ├── category_info.json (emotion category definitions)"
echo "    ├── emotion_categories.png (2D scatter plot of categories)"
echo "    ├── category_distribution.png (bar chart of category counts)"
echo "    └── [category_name]/ (steering signals for each category)"
echo "      ├── valence_128d.npy"
echo "      ├── arousal_128d.npy"
echo "      └── ... (other activation types)"
echo ""
echo "  📁 $TEST_RESULTS_DIR/"
echo "    ├── steering_test_results.json (detailed test results)"
echo "    ├── steering_test_summary.txt (statistical summary)"
echo "    ├── valence_changes.png (box plot of valence changes)"
echo "    ├── arousal_changes.png (box plot of arousal changes)"
echo "    └── steering_effects_2d.png (2D scatter of steering effects)"
echo ""
echo "Next Steps:"
echo "  1. Examine the visualizations to understand steering effects"
echo "  2. Use the steering signals in your own experiments"
echo "  3. Modify the categorization method or add new steering signals"
echo ""
echo "To use steering signals in your own code:"
echo "  # Load steering signals"
echo "  steering_signals, category_info = load_steering_signals('$STEERING_DIR')"
echo "  "
echo "  # Apply to model"
echo "  valence_128d = torch.from_numpy(steering_signals['positive_strong']['valence_128d'])"
echo "  arousal_128d = torch.from_numpy(steering_signals['positive_strong']['arousal_128d'])"
echo "  model.set_external_feedback(valence_128d=valence_128d, arousal_128d=arousal_128d)" 