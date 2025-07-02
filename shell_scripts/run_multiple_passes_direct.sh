#!/bin/bash

# Simple script to run multiple passes testing with the specific model path
# No interactive prompts - just runs the test directly

echo "🚀 Running Multiple Passes Test (All Validation Samples)"
echo "======================================================"

# Model path (as provided by user)
MODEL_PATH="workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=16/freeze_base=True/best_model.pth"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model not found at: $MODEL_PATH"
    echo "Please check the path and try again."
    exit 1
fi

echo "✅ Model found at: $MODEL_PATH"
echo ""

# Run the test with all validation samples
echo "🔬 Running multiple passes test with ALL validation samples..."
echo "This will test the model with the complete validation set for comprehensive analysis."
echo ""

bash shell_scripts/test_multiple_passes.sh \
    --model-path "$MODEL_PATH" \
    --max-passes 6 \
    --batch-size 16 \
    --num-samples 0

echo ""
echo "🎉 Multiple passes testing completed!"
echo ""
echo "📁 Results saved in: workspaces/emotion_feedback/multiple_passes_test/"
echo ""
echo "💡 To view results:"
echo "  - Check the console output above for summary"
echo "  - Open the generated plots in the results directory"
echo "  - Review the CSV/JSON files for detailed analysis" 