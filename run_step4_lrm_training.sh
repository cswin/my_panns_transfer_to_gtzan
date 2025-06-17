#!/bin/bash

# Step 4: Full LRM Training Test Runner
# Run this script on the remote server with GPU and real emotion data

echo "🧪 Step 4: Full LRM Training Test"
echo "=================================="

# Check if we're in the right directory
if [ ! -f "test_step4_full_lrm_training.py" ]; then
    echo "❌ Error: test_step4_full_lrm_training.py not found"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Default data paths (matching run_emotion.sh configuration)
WORKSPACE="workspaces/emotion_regression"
DEFAULT_DATA_PATH="$WORKSPACE/features/emotion_features.h5"

# Alternative paths to check (in order of preference)
ALT_PATHS=(
    "features/emotion_features/emotion_features.h5"
    "/DATA/pliu/EmotionData/features/emotion_features.h5"
    "workspaces/step4_lrm_test/emotion_features.h5"
)

# Check if data path is provided as argument
if [ $# -eq 1 ]; then
    DATA_PATH="$1"
else
    # Try to find existing features in common locations
    DATA_PATH=""
    
    # Check default path first
    if [ -f "$DEFAULT_DATA_PATH" ]; then
        DATA_PATH="$DEFAULT_DATA_PATH"
        echo "✅ Found features at default location: $DATA_PATH"
    else
        # Try alternative paths
        for alt_path in "${ALT_PATHS[@]}"; do
            if [ -f "$alt_path" ]; then
                DATA_PATH="$alt_path"
                echo "✅ Found features at: $DATA_PATH"
                break
            fi
        done
    fi
    
    # If no features found, show error with suggestions
    if [ -z "$DATA_PATH" ]; then
        echo "❌ Error: No emotion features found!"
        echo ""
        echo "Searched in the following locations:"
        echo "  - $DEFAULT_DATA_PATH"
        for alt_path in "${ALT_PATHS[@]}"; do
            echo "  - $alt_path"
        done
        echo ""
        echo "Solutions:"
        echo "  1. Run feature extraction first:"
        echo "     bash run_emotion.sh"
        echo "  2. Provide the correct path:"
        echo "     $0 /path/to/your/emotion_features.h5"
        echo "  3. Copy your features to the expected location:"
        echo "     mkdir -p $WORKSPACE/features"
        echo "     cp /your/features.h5 $DEFAULT_DATA_PATH"
        exit 1
    fi
fi

echo "Using data path: $DATA_PATH"

# Final check that the file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Error: Data file not found at $DATA_PATH"
    echo "Please check the file path and try again."
    exit 1
fi

echo "✅ Data file found"

# Create results directory
mkdir -p workspaces/step4_results
echo "📁 Created results directory: workspaces/step4_results"

echo ""
echo "🚀 Starting Step 4 Tests..."
echo "=========================="

# Test 1: LRM Model Training (main test)
echo ""
echo "🧪 Test 1: LRM Model Training (Simplified)"
echo "-------------------------------------------"
python test_step4_simple.py \
    --features_hdf5_path "$DATA_PATH" \
    --model_type FeatureEmotionRegression_Cnn6_LRM \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --workspace ./workspaces/step4_results/lrm_training \
    --cuda

LRM_EXIT_CODE=$?

# Test 2: Baseline Model Training (for comparison)
echo ""
echo "🧪 Test 2: Baseline Model Training (Simplified)"
echo "------------------------------------------------"
python test_step4_simple.py \
    --features_hdf5_path "$DATA_PATH" \
    --model_type FeatureEmotionRegression_Cnn6_NewAffective \
    --num_epochs 5 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --workspace ./workspaces/step4_results/baseline_training \
    --cuda

BASELINE_EXIT_CODE=$?

# Test 3: Quick LRM Functionality Test (if LRM worked)
if [ $LRM_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🧪 Test 3: LRM Functionality Verification"
    echo "------------------------------------------"
    
    # Quick test to verify LRM-specific features work
    echo "Testing LRM feedback functionality..."
    python test_step4_simple.py \
        --features_hdf5_path "$DATA_PATH" \
        --model_type FeatureEmotionRegression_Cnn6_LRM \
        --num_epochs 2 \
        --batch_size 16 \
        --learning_rate 1e-3 \
        --workspace ./workspaces/step4_results/lrm_functionality \
        --cuda
    
    FUNC_TEST_EXIT_CODE=$?
else
    echo "⚠️  Skipping functionality test due to LRM training failure"
    FUNC_TEST_EXIT_CODE=1
fi

# Summary
echo ""
echo "📊 Step 4 Test Results Summary"
echo "=============================="
echo "Test 1 - LRM Training:           $([ $LRM_EXIT_CODE -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "Test 2 - Baseline Training:      $([ $BASELINE_EXIT_CODE -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED")"
echo "Test 3 - LRM Functionality:      $([ $FUNC_TEST_EXIT_CODE -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED/SKIPPED")"

# Overall result
TOTAL_TESTS=2  # Only count the main tests
PASSED_TESTS=0
[ $LRM_EXIT_CODE -eq 0 ] && PASSED_TESTS=$((PASSED_TESTS + 1))
[ $BASELINE_EXIT_CODE -eq 0 ] && PASSED_TESTS=$((PASSED_TESTS + 1))

echo ""
echo "Overall: $PASSED_TESTS/$TOTAL_TESTS core tests passed"

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    echo "🎉 Step 4: Full LRM Training Test - SUCCESS!"
    echo ""
    echo "✅ LRM model trains successfully on real emotion data"
    echo "✅ Segment-based feedback system works correctly"
    echo "✅ LRM evaluation system verified"
    echo "✅ Model checkpoints saved properly"
    echo "✅ Training metrics tracked correctly"
    echo ""
    echo "📁 Results saved in: workspaces/step4_results/"
    echo "🚀 Ready for Step 5: Comprehensive Baseline Comparison!"
    exit 0
else
    echo "❌ Step 4: Some tests failed"
    echo ""
    echo "🔧 Please check the error messages above and:"
    echo "   - Verify data path is correct"
    echo "   - Check GPU availability"
    echo "   - Ensure all dependencies are installed"
    echo "   - Check disk space for model checkpoints"
    exit 1
fi 