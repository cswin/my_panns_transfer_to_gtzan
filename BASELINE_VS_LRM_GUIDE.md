# Baseline vs LRM Comparison Guide

This guide shows how to run a complete comparison between the baseline emotion model and the LRM (Long Range Modulation) model with feedback.

## 🗂️ **Separate Workspaces (No Overwriting)**

The scripts automatically save results in **separate directories**:
- **Baseline Model**: `workspaces/emotion_regression/`
- **LRM Model**: `workspaces/emotion_feedback/`

This ensures no results are overwritten when running both models.

## 🚀 **Quick Start Options**

### Option 1: Complete Comparison (Recommended)
```bash
# Run both models and generate comparison plots
bash run_baseline_vs_lrm_comparison.sh
```

### Option 2: Step-by-Step
```bash
# Step 1: Check current status
python check_workspaces.py

# Step 2: Run baseline model
./run_emotion.sh --skip-extraction

# Step 3: Run LRM model  
./run_emotion_feedback.sh --skip-extraction

# Step 4: Generate comparison plots
python plot_training_comparison.py --auto-find
```

### Option 3: Selective Training
```bash
# Only baseline (if LRM already done)
bash run_baseline_vs_lrm_comparison.sh --skip-lrm

# Only LRM (if baseline already done)  
bash run_baseline_vs_lrm_comparison.sh --skip-baseline

# Only plots (if both trainings done)
bash run_baseline_vs_lrm_comparison.sh --plots-only
```

## 📊 **What Gets Generated**

### Training Results
```
workspaces/
├── emotion_regression/          # Baseline model
│   ├── checkpoints/            # Model checkpoints
│   ├── logs/                   # Training logs  
│   ├── statistics/             # Training statistics
│   └── predictions/            # Evaluation results
└── emotion_feedback/           # LRM model
    ├── checkpoints/            # Model checkpoints
    ├── logs/                   # Training logs
    ├── statistics/             # Training statistics  
    └── predictions/            # Evaluation results
```

### Comparison Plots
```
training_comparison_plots/
├── training_comparison.png     # Main 4-panel overview
├── train_loss_comparison.png   # Training loss over epochs
├── val_loss_comparison.png     # Validation loss over epochs
├── val_mae_comparison.png      # Validation MAE over epochs
└── val_pearson_comparison.png  # Validation correlation over epochs
```

## 🔍 **Model Differences**

| Aspect | Baseline Model | LRM Model |
|--------|---------------|-----------|
| **Architecture** | FeatureEmotionRegression_Cnn6_NewAffective | FeatureEmotionRegression_Cnn6_LRM |
| **Feedback** | None (feedforward only) | Top-down feedback connections |
| **Forward Passes** | 1 | 2 (iterative refinement) |
| **Psychological Motivation** | Standard CNN processing | Arousal→attention, Valence→semantics |
| **Workspace** | `workspaces/emotion_regression/` | `workspaces/emotion_feedback/` |

## 📈 **Expected LRM Advantages**

The LRM model should demonstrate:
- **Lower validation loss** (better generalization)
- **Lower validation MAE** (more accurate emotion predictions)
- **Higher Pearson correlation** (better linear relationship with ground truth)
- **Faster convergence** (reaches good performance in fewer epochs)

## 🛠️ **Troubleshooting**

### Check Status
```bash
python check_workspaces.py
```

### Manual Plot Generation
```bash
# Auto-find logs
python plot_training_comparison.py --auto-find

# Specify log files directly
python plot_training_comparison.py \
    --baseline-logs workspaces/emotion_regression/logs/train.log \
    --lrm-logs workspaces/emotion_feedback/logs/train.log
```

### View Demo Plots
```bash
# See what plots look like with sample data
python run_training_plots.py --demo
```

## 📝 **Command Reference**

| Command | Purpose |
|---------|---------|
| `bash run_baseline_vs_lrm_comparison.sh` | Complete comparison |
| `bash run_baseline_vs_lrm_comparison.sh --help` | Show help |
| `python check_workspaces.py` | Check current status |
| `python plot_training_comparison.py --auto-find` | Generate plots |
| `python run_training_plots.py --demo` | Demo plots |

## 🎯 **Next Steps After Training**

1. **View main comparison**: Open `training_comparison_plots/training_comparison.png`
2. **Analyze individual metrics**: Check separate metric plots
3. **Review training logs**: Examine detailed logs in each workspace
4. **Compare final metrics**: Look at the summary statistics
5. **Document results**: Save plots and metrics for your research

## 🔬 **Research Applications**

This comparison validates:
- **Psychological feedback principles** in neural networks
- **Top-down modulation effectiveness** for emotion recognition  
- **Iterative refinement benefits** in audio processing
- **Attention mechanisms** for acoustic feature processing

Perfect for research papers, conference presentations, and further model development! 