# Multiple Passes Testing for Emotion Feedback Model

This document explains how to test the emotion feedback model with different numbers of forward passes to analyze performance as a function of the number of feed-forward iterations.

## Overview

The emotion feedback model (`FeatureEmotionRegression_Cnn6_LRM`) supports multiple forward passes where feedback signals from previous passes can modulate subsequent passes. This testing framework allows you to:

- Test performance with 1 to N forward passes
- Measure performance metrics (MAE, RMSE, Pearson correlation) for each pass count
- Analyze computational cost vs performance trade-offs
- Visualize performance trends with comprehensive plots

## Files Created

### Main Scripts
- `shell_scripts/test_multiple_passes.sh` - Main bash script for comprehensive testing
- `scripts/test_multiple_passes.py` - Full Python implementation with real dataset
- `scripts/test_multiple_passes_simple.py` - Simple test with dummy data for quick validation

### Output Files
- `multiple_passes_results.csv` - Detailed results table
- `multiple_passes_results.json` - Results in JSON format
- `multiple_passes_performance.png` - Comprehensive performance plots
- `multiple_passes_summary.png` - Summary plot with performance vs cost

## Usage

### Quick Start (Simple Test)

For a quick test with dummy data to verify the setup:

```bash
# Test with default settings (1-6 passes, 50 dummy samples)
bash shell_scripts/test_multiple_passes.sh --model-path /path/to/your/model.pth

# Test with custom settings
bash shell_scripts/test_multiple_passes.sh \
    --model-path /path/to/your/model.pth \
    --max-passes 8 \
    --num-samples 100
```

### Full Evaluation (Real Dataset)

For comprehensive evaluation with real emotion dataset:

```bash
# Full evaluation with auto-detected paths
bash shell_scripts/test_multiple_passes.sh \
    --model-path /path/to/your/model.pth \
    --dataset-path /path/to/emotion_features.h5

# Full evaluation with custom settings
bash shell_scripts/test_multiple_passes.sh \
    --model-path /path/to/your/model.pth \
    --dataset-path /path/to/emotion_features.h5 \
    --max-passes 10 \
    --batch-size 32 \
    --num-samples 200 \
    --workspace workspaces/emotion_feedback
```

### Direct Python Usage

You can also run the Python scripts directly:

```bash
# Simple test
python3 scripts/test_multiple_passes_simple.py \
    --model-path /path/to/your/model.pth \
    --max-passes 6 \
    --num-samples 50

# Full evaluation
python3 scripts/test_multiple_passes.py \
    --model-path /path/to/your/model.pth \
    --dataset-path /path/to/emotion_features.h5 \
    --max-passes 6 \
    --batch-size 16 \
    --num-samples 100 \
    --output-dir results/multiple_passes_test
```

## Command Line Options

### Bash Script Options
- `--max-passes N` - Maximum number of passes to test (default: 6)
- `--model-path PATH` - Path to specific model checkpoint
- `--workspace PATH` - Workspace directory (default: workspaces/emotion_feedback)
- `--dataset-path PATH` - Path to emotion features HDF5 file
- `--batch-size N` - Batch size for evaluation (default: 16)
- `--num-samples N` - Number of samples to test per pass (default: 100)
- `--help` - Show help message

### Python Script Options
- `--model-path` - Path to model checkpoint (required)
- `--dataset-path` - Path to emotion features HDF5 file (required for full script)
- `--max-passes` - Maximum number of passes to test (default: 6)
- `--batch-size` - Batch size for evaluation (default: 16)
- `--num-samples` - Number of samples to test per pass (default: 100)
- `--output-dir` - Output directory for results (default: results)

## Auto-Detection

The bash script automatically detects common paths:

### Model Path Auto-Detection
1. `$WORKSPACE/checkpoints/best_model.pth`
2. Most recent model file in `$WORKSPACE/checkpoints/model_*.pth`

### Dataset Path Auto-Detection
1. `$WORKSPACE/features/emotion_features.h5`
2. `workspaces/emotion_regression/features/emotion_features.h5`
3. `features/emotion_features/emotion_features.h5`

## Output Analysis

### Performance Metrics
For each number of passes, the script measures:
- **Valence MAE/RMSE/Pearson/Spearman** - Valence prediction accuracy
- **Arousal MAE/RMSE/Pearson/Spearman** - Arousal prediction accuracy
- **Mean MAE/RMSE/Pearson/Spearman** - Average across valence and arousal
- **Processing Time** - Computational cost per sample

### Key Insights to Look For

1. **Optimal Number of Passes**: Where performance peaks
2. **Performance Saturation**: When additional passes don't improve performance
3. **Computational Trade-off**: Performance gain vs processing time cost
4. **Feedback Effectiveness**: Whether multiple passes actually improve performance

### Visualization

The script generates two main plots:

1. **Comprehensive Performance Plot** (`multiple_passes_performance.png`):
   - 6 subplots showing different metrics vs pass count
   - Separate valence/arousal performance
   - Processing time analysis

2. **Summary Plot** (`multiple_passes_summary.png`):
   - Combined view of performance vs computational cost
   - Dual y-axis showing metrics and processing time

## Example Output

```
🧪 Testing Multiple Passes for Emotion Feedback Model
==================================================

Configuration:
  - Max Passes: 6
  - Model Path: workspaces/emotion_feedback/checkpoints/best_model.pth
  - Dataset Path: workspaces/emotion_feedback/features/emotion_features.h5
  - Workspace: workspaces/emotion_feedback
  - Batch Size: 16
  - Samples per Test: 100

🔬 Testing multiple passes (1 to 6)
📊 Using 100 samples per test
============================================================
Using device: cuda
Loading model...
Loading dataset...
Collecting test data...

🔄 Testing 1 forward pass(es)...
  ✅ 1 pass(es) - Mean MAE: 0.2345, Mean Pearson: 0.7123
      Valence: MAE=0.2234, Pearson=0.7234
      Arousal: MAE=0.2456, Pearson=0.7012
      Avg time: 45.23ms

🔄 Testing 2 forward pass(es)...
  ✅ 2 pass(es) - Mean MAE: 0.2123, Mean Pearson: 0.7456
      Valence: MAE=0.2012, Pearson=0.7567
      Arousal: MAE=0.2234, Pearson=0.7345
      Avg time: 89.45ms

...

📊 MULTIPLE PASSES TESTING SUMMARY
============================================================
🎯 Best MAE: 0.1987 at 3 passes
🎯 Best Pearson: 0.7567 at 3 passes

📈 Performance vs Passes:
  1 pass(es): MAE=0.2345, Pearson=0.7123, Time=45.2ms
  2 pass(es): MAE=0.2123, Pearson=0.7456, Time=89.5ms
  3 pass(es): MAE=0.1987, Pearson=0.7567, Time=134.1ms
  4 pass(es): MAE=0.2012, Pearson=0.7543, Time=178.9ms
  5 pass(es): MAE=0.2034, Pearson=0.7521, Time=223.4ms
  6 pass(es): MAE=0.2056, Pearson=0.7502, Time=267.8ms
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model path is correct or let auto-detection work
2. **Dataset not found**: Check dataset path or run feature extraction first
3. **CUDA out of memory**: Reduce batch size or number of samples
4. **Import errors**: Ensure all dependencies are installed and PYTHONPATH is set

### Performance Tips

1. **For quick testing**: Use the simple script with dummy data
2. **For accurate results**: Use the full script with real dataset
3. **For faster execution**: Reduce number of samples or max passes
4. **For better visualization**: Increase number of samples for smoother curves

## Integration with Existing Workflow

This testing framework integrates with the existing emotion feedback pipeline:

1. **After training**: Use to find optimal number of passes for your model
2. **Before deployment**: Validate performance vs computational cost
3. **For research**: Analyze feedback effectiveness across different conditions
4. **For comparison**: Compare different model variants with same testing protocol

## Future Enhancements

Potential improvements for the testing framework:

1. **Steering signal testing**: Test with different steering signal strengths
2. **Model comparison**: Compare feedback vs non-feedback models
3. **Statistical significance**: Add confidence intervals and significance testing
4. **Interactive plots**: Create interactive visualizations for exploration
5. **Batch processing**: Support for testing multiple models simultaneously 