#!/bin/bash

# Script to demonstrate 25-bin external steering signals for emotion feedback testing
# This script shows how to:
# 1. Generate 25-bin steering signals by categorizing audio data into 5x5 valence/arousal grid
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
STEERING_DIR="steering_signals_25bin"
TEST_RESULTS_DIR="steering_test_results_25bin"

# =============================================================================
# Validation Checks
# =============================================================================

echo "🔄 Setting up 25-Bin External Steering Signals Pipeline..."
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
# Step 1: Generate 25-Bin Steering Signals
# =============================================================================

echo "Step 1: Generating 25-Bin External Steering Signals"
echo "🎯 Categorizing audio data into 25 emotion bins (5x5 valence/arousal grid)"
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
    echo "🔄 Running 25-bin steering signal generation..."
    
    PYTHONPATH=. python3 scripts/generate_25bin_steering_signals.py \
        --dataset_path "$DATASET_PATH" \
        --model_checkpoint "$MODEL_CHECKPOINT" \
        --output_dir "$STEERING_DIR" \
        --batch_size 32 \
        --cuda
    
    if [ $? -ne 0 ]; then
        echo "Error: 25-bin steering signal generation failed!"
        exit 1
    fi
    
    echo "✅ 25-bin steering signals generated successfully!"
    echo "📁 Output directory: $STEERING_DIR"
    echo "📊 Generated visualizations: 25bin_steering_signals_analysis.png"
    echo ""
fi

# =============================================================================
# Step 2: Test 25-Bin Steering Signals
# =============================================================================

echo "Step 2: Testing 25-Bin Steering Signals with Emotion Model"
echo "🧪 Applying steering signals to test samples and measuring effects"
echo ""

echo "🔄 Running 25-bin steering signal testing..."
PYTHONPATH=src:. python3 scripts/experimental/test_25bin_comprehensive.py \
    --dataset_path "$DATASET_PATH" \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --steering_9bin "steering_signals/steering_signals_by_category.json" \
    --steering_25bin "$STEERING_DIR/steering_signals_25bin.json" \
    --num_samples 100000 \
    --cuda

if [ $? -ne 0 ]; then
    echo "Error: 25-bin steering signal testing failed!"
    exit 1
fi

echo "✅ 25-bin steering signal testing completed!"
echo "📁 Results displayed in terminal"
echo ""

# =============================================================================
# Summary
# =============================================================================

echo "🎉 25-Bin External Steering Signals Pipeline Completed Successfully!"
echo ""
echo "Summary:"
echo "  - Generated 25 steering signal categories based on 5x5 valence/arousal grid"
echo "  - Tested steering effects on all validation samples"
echo "  - Compared 25-bin vs 9-bin categorical methods"
echo "  - Created visualizations of steering signal impacts"
echo ""
echo "Key Files Generated:"
echo "  📁 $STEERING_DIR/"
echo "    ├── steering_signals_25bin.json (25-bin steering signals)"
echo "    ├── 25bin_steering_signals_analysis.png (analysis visualization)"
echo "    └── [category_name]/ (steering signals for each category)"
echo "      ├── valence_128d.npy"
echo "      ├── arousal_128d.npy"
echo "      └── ... (other activation types)"
echo ""
echo "Next Steps:"
echo "  1. Examine the terminal output for performance comparison"
echo "  2. Compare results with 9-bin steering signals"
echo "  3. Use the 25-bin steering signals in your own experiments"
echo ""
echo "To use 25-bin steering signals in your own code:"
echo "  # Load 25-bin steering signals"
echo "  with open('$STEERING_DIR/steering_signals_25bin.json', 'r') as f:"
echo "      steering_signals = json.load(f)"
echo "  "
echo "  # Apply to model (example: very_positive_strong category)"
echo "  valence_128d = torch.tensor(steering_signals['very_positive_strong']['valence_128d'])"
echo "  arousal_128d = torch.tensor(steering_signals['very_positive_strong']['arousal_128d'])"
echo "  model.add_steering_signal(source='affective_valence_128d', activation=valence_128d, strength=5.0)"
echo "  model.add_steering_signal(source='affective_arousal_128d', activation=arousal_128d, strength=5.0)" 