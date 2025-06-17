# Step 4: Full LRM Training Test

## Overview
Step 4 runs comprehensive training tests with the LRM (Long Range Modulation) emotion model on real Emo-Soundscapes data. This validates that our LRM implementation works correctly in a full training environment with proper segment-based feedback.

## What Step 4 Tests

### ✅ Core Functionality
1. **LRM Model Training**: Full training session with real emotion data
2. **Segment-based Feedback**: Verifies feedback system works during training
3. **LRM Evaluation System**: Tests our custom LRM-aware evaluation
4. **Model Checkpointing**: Saves and loads trained models
5. **Training Metrics**: Tracks loss, early stopping, etc.

### 🔬 Experimental Validation
- **Baseline Comparison**: Trains both LRM and non-feedback models
- **Modulation Strength**: Tests different feedback strengths (0.5, 1.0, 2.0)
- **Performance Tracking**: Monitors training curves and convergence

## Files Created

### Main Test Script
- **`test_step4_full_lrm_training.py`**: Complete training test with argument parsing
  - Supports both LRM and baseline models
  - Configurable training parameters
  - Automatic model saving and evaluation
  - Comprehensive error handling

### Runner Script
- **`run_step4_lrm_training.sh`**: Easy-to-use shell script for remote server
  - Automatic data path detection
  - Multiple test configurations
  - Results summary and validation
  - Error checking and reporting

## Usage on Remote Server

### Quick Start
```bash
# Make executable (if needed)
chmod +x run_step4_lrm_training.sh

# Run with default data path
./run_step4_lrm_training.sh

# Run with custom data path
./run_step4_lrm_training.sh /path/to/your/emo_soundscapes_features.h5
```

### Manual Usage
```bash
# LRM model training
python test_step4_full_lrm_training.py \
    --features_hdf5_path /data/emo_soundscapes/features/emo_soundscapes_features.h5 \
    --model_type FeatureEmotionRegression_Cnn6_LRM \
    --num_epochs 15 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --modulation_strength 1.0 \
    --workspace ./workspaces/lrm_training \
    --cuda --freeze_base --save_model

# Baseline model training
python test_step4_full_lrm_training.py \
    --model_type FeatureEmotionRegression_Cnn6_NewAffective \
    --workspace ./workspaces/baseline_training \
    # ... other args same as above
```

## Expected Results

### ✅ Success Indicators
1. **Model Creation**: Both LRM and baseline models initialize correctly
2. **Training Progress**: Loss decreases over epochs
3. **LRM Feedback**: Segment-based feedback affects predictions
4. **Evaluation**: LRM-aware evaluation system works
5. **Checkpoints**: Models save and load successfully

### 📊 Performance Metrics
- **Training Loss**: Should decrease over epochs
- **Validation Loss**: Should improve with early stopping
- **LRM Evaluation**: Valence/Arousal MAE on sequential segments
- **Parameter Count**: LRM has ~95k additional parameters

### 🔍 What to Look For
- **Feedback Effects**: LRM predictions should change over segments
- **Training Stability**: No gradient explosions or NaN losses
- **Memory Usage**: Reasonable GPU memory consumption
- **Convergence**: Models should reach stable performance

## Troubleshooting

### Common Issues
1. **Data Path**: Ensure HDF5 features file exists
2. **GPU Memory**: Reduce batch size if OOM errors
3. **Dependencies**: Check pytorch, h5py, numpy versions
4. **Disk Space**: Ensure space for model checkpoints

### Debug Commands
```bash
# Check data file
ls -la /data/emo_soundscapes/features/emo_soundscapes_features.h5

# Check GPU
nvidia-smi

# Test imports
python -c "import torch; print(torch.cuda.is_available())"
```

## Architecture Validation

### LRM Components Tested
1. **ModBlock**: Emotion → modulation signal transformation
2. **LongRangeModulation**: Forward hook system
3. **Feedback Connections**: 4 psychologically-motivated paths
4. **Sequential Processing**: Segment-based feedback accumulation

### Psychological Connections
- **Arousal → conv1, conv2**: Energy affects attention to details
- **Valence → conv3, conv4**: Emotion affects semantic processing

## Next Steps

After Step 4 success:
1. **Step 5**: Comprehensive baseline comparison
2. **Analysis**: Compare LRM vs non-feedback performance
3. **Optimization**: Tune modulation strengths and architectures
4. **Publication**: Document findings and improvements

## Results Location
- **Workspaces**: `./workspaces/step4_results/`
- **Models**: `best_*.pth` checkpoint files
- **Metrics**: `training_history.npz` files
- **Logs**: Console output with training progress

---

**Status**: ✅ Ready for execution on remote server
**Dependencies**: Real Emo-Soundscapes data, GPU environment
**Duration**: ~30-60 minutes depending on data size and epochs 