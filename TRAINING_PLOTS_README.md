# Training Performance Comparison Tools

This directory contains tools to create training performance plots comparing baseline (`run_emotion.sh`) and LRM (`run_emotion_feedback.sh`) models.

## 📊 Available Tools

### 1. `plot_training_comparison.py` - Main Plotting Tool
Advanced plotting tool that can parse different log formats and create comprehensive visualizations.

### 2. `run_training_plots.py` - Simple Usage Script  
Easy-to-use script that demonstrates the plotting functionality with both demo data and real logs.

## 🚀 Quick Start

### Option 1: Demo Plots (No training required)
```bash
python run_training_plots.py --demo
```
This creates sample plots showing what the comparison would look like.

### Option 2: Auto-find Real Logs
```bash
python run_training_plots.py --real
```
This searches for training logs and creates plots from real data.

### Option 3: Advanced Usage
```bash
# Auto-find logs in current directory
python plot_training_comparison.py --auto-find

# Specify log files directly
python plot_training_comparison.py \
    --baseline-logs workspaces/emotion_regression/logs/baseline_train.log \
    --lrm-logs workspaces/emotion_regression/logs/lrm_train.log

# Search in specific directories
python plot_training_comparison.py --auto-find --search-dirs ./workspaces ./logs
```

## 📈 Generated Plots

The tools create several types of plots:

### 1. Main Comparison Plot (`training_comparison.png`)
4-panel comparison showing:
- Training Loss over epochs
- Validation Loss over epochs  
- Validation MAE over epochs
- Validation Pearson Correlation over epochs

### 2. Individual Metric Plots
Separate detailed plots for each metric:
- `train_loss_comparison.png`
- `val_loss_comparison.png`
- `val_mae_comparison.png`
- `val_rmse_comparison.png`
- `val_pearson_comparison.png`

## 📋 Supported Log Formats

The parser automatically detects and handles:

### 1. Emotion Main Format (`emotion_main.py`)
```
Iteration: 200, loss: 1.234
Validate Audio Mean MAE: 0.567
Validate Audio Mean RMSE: 0.789
Validate Audio Mean Pearson: 0.456
```

### 2. Epoch-Based Format (Test Scripts)
```
Epoch 1 Training: Loss=1.234 (V=0.567, A=0.890)
Epoch 1 Validation: Loss=1.345 (V=0.678, A=0.901)
```

## 🎯 How to Run Baseline vs LRM Comparison

### Step 1: Train Baseline Model
```bash
./run_emotion.sh --skip-extraction
```

### Step 2: Train LRM Model  
```bash
./run_emotion_feedback.sh --skip-extraction
```

### Step 3: Generate Comparison Plots
```bash
# Auto-find and plot both
python plot_training_comparison.py --auto-find

# Or specify log files directly
python plot_training_comparison.py \
    --baseline-logs <path_to_baseline_log> \
    --lrm-logs <path_to_lrm_log>
```

## 🔍 Expected Results

The LRM model should show:
- **Lower validation loss** (better overall performance)
- **Lower validation MAE** (better emotion prediction accuracy)
- **Higher Pearson correlation** (better linear relationship with ground truth)
- **Faster convergence** (reaches good performance in fewer epochs)

## 📊 Interpreting the Plots

### Training Loss
- Shows how well the model learns during training
- Both models should decrease over time
- LRM may show faster initial decrease due to feedback

### Validation Loss  
- Most important metric for comparing models
- Lower is better
- Shows generalization performance

### Validation MAE (Mean Absolute Error)
- Direct measure of emotion prediction accuracy
- Lower values = better predictions
- Typical good values: < 0.3 for normalized emotions

### Validation Pearson Correlation
- Measures linear relationship with ground truth
- Higher values = better correlation
- Good values: > 0.7 for emotion prediction

## 🛠️ Troubleshooting

### No Log Files Found
```bash
# Check if logs exist
find . -name "*.log" -type f

# Check workspaces directory
ls -la workspaces/emotion_regression/logs/
```

### Parsing Errors
- Ensure log files contain training metrics
- Check log format matches expected patterns
- Try with `--auto-find` to let the tool detect format

### Missing Dependencies
```bash
pip install matplotlib pandas numpy
```

## 📝 Example Output

```
🚀 Training Performance Comparison Tool
==================================================
🔍 Auto-finding log files...
Found baseline logs: ['./workspaces/emotion_regression/logs/train.log']
Found LRM logs: ['./workspaces/emotion_regression/logs/lrm_train.log']
✅ Parsed baseline log: ./workspaces/emotion_regression/logs/train.log
✅ Parsed LRM log: ./workspaces/emotion_regression/logs/lrm_train.log
📈 Creating training comparison plots...
✅ Plots saved in: ./training_plots

📊 Training Summary:
----------------------------------------
Baseline final training loss: 0.756
LRM final training loss: 0.463
MAE improvement: +37.8% (Baseline: 0.450, LRM: 0.280)
```

## 🎨 Customization

You can modify the plotting colors and styles in `plot_training_comparison.py`:

```python
self.colors = {
    'baseline': '#2E86AB',  # Blue
    'lrm': '#A23B72',       # Purple
    'train': '#F18F01',     # Orange
    'val': '#C73E1D'        # Red
}
```

## 📞 Support

If you encounter issues:
1. Check that training logs contain the expected metrics
2. Verify log file paths are correct
3. Try the demo mode first to ensure plotting works
4. Check that all dependencies are installed

The tools are designed to be robust and handle various log formats automatically! 