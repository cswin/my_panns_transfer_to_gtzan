#!/bin/bash

# Script to test steering signals vs internal feedback
# This script demonstrates how the model automatically switches between external steering and internal feedback

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEST_SCRIPT="$PROJECT_DIR/scripts/debug/test_steering_vs_internal.py"

# Default values
MODEL_PATH=""
DATASET_PATH=""
NUM_SAMPLES=2

# Function to print usage
print_usage() {
    echo "Usage: $0 --model-path <path> --dataset-path <path> [--num-samples <number>]"
    echo ""
    echo "Arguments:"
    echo "  --model-path <path>      Path to the emotion feedback model checkpoint"
    echo "  --dataset-path <path>    Path to the emotion features HDF5 file"
    echo "  --num-samples <number>   Number of samples to test (default: 2)"
    echo ""
    echo "Example:"
    echo "  $0 --model-path /path/to/model.pth --dataset-path /path/to/features.h5"
    echo ""
    echo "This test demonstrates:"
    echo "  1. Internal feedback (model's own predictions) - currently broken"
    echo "  2. External steering signals - should work"
    echo "  3. Legacy external feedback - should work"
    echo "  4. Priority system implementation"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "❌ Error: --model-path is required"
    print_usage
    exit 1
fi

if [[ -z "$DATASET_PATH" ]]; then
    echo "❌ Error: --dataset-path is required"
    print_usage
    exit 1
fi

# Check if files exist
if [[ ! -f "$MODEL_PATH" ]]; then
    echo "❌ Error: Model file not found: $MODEL_PATH"
    exit 1
fi

if [[ ! -f "$DATASET_PATH" ]]; then
    echo "❌ Error: Dataset file not found: $DATASET_PATH"
    exit 1
fi

# Check if test script exists
if [[ ! -f "$TEST_SCRIPT" ]]; then
    echo "❌ Error: Test script not found: $TEST_SCRIPT"
    exit 1
fi

echo "🧪 Testing Steering Signals vs Internal Feedback"
echo "==============================================="
echo "Model path: $MODEL_PATH"
echo "Dataset path: $DATASET_PATH"
echo "Number of samples: $NUM_SAMPLES"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Activate virtual environment if it exists
if [[ -d "venv" ]]; then
    echo "🐍 Activating virtual environment..."
    source venv/bin/activate
fi

# Run the steering vs internal feedback test
echo "🚀 Running steering vs internal feedback test..."
python "$TEST_SCRIPT" \
    --model-path "$MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --num-samples "$NUM_SAMPLES"

echo ""
echo "✅ Steering vs internal feedback test completed!"
echo ""
echo "📋 Expected Results:"
echo "   - Internal feedback: ❌ Should NOT work (hooks not registered)"
echo "   - External steering: ✅ Should work (manual signal injection)"
echo "   - Legacy feedback:   ✅ Should work (manual signal injection)"
echo "   - Priority system:   ✅ Should be correctly implemented" 