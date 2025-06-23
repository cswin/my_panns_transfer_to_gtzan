# Project Structure Overview

This document shows the reorganized structure of the GTZAN Music Genre Classification project.

## New Directory Structure

```
my_panns_transfer_to_gtzan/
├── src/                          # Main source code
│   ├── __init__.py               # Package initialization
│   ├── models/                   # Neural network models
│   │   ├── __init__.py           # Models package init
│   │   ├── cnn_models.py         # CNN-based models (Cnn6, Cnn14)
│   │   ├── emotion_models.py     # Emotion regression models
│   │   └── losses.py             # Loss functions
│   ├── data/                     # Data processing
│   │   ├── __init__.py           # Data package init
│   │   ├── data_generator.py     # Data loading and batching
│   │   ├── feature_extractor.py  # Audio feature extraction
│   │   └── dataset_utils.py      # Dataset utilities
│   ├── training/                 # Training scripts
│   │   ├── __init__.py           # Training package init
│   │   ├── trainer.py            # Main training loop
│   │   ├── evaluator.py          # Model evaluation
│   │   ├── evaluator_lrm.py      # Emotion-specific evaluation
│   │   └── config.py             # Training configuration
│   └── utils/                    # Utility functions
│       ├── __init__.py           # Utils package init
│       ├── audio_utils.py        # Audio processing utilities
│       ├── config.py             # Utility configurations
│       ├── pytorch_utils.py      # PyTorch utilities
│       ├── visualization.py      # Plotting and visualization
│       └── visualization_latest.py # Latest visualization tools
├── scripts/                      # Executable scripts
│   ├── train.py                  # Main training script
│   ├── evaluate.py               # Model evaluation script
│   ├── extract_features.py       # Feature extraction script
│   └── test_models.py            # Model testing script
├── examples/                     # Example usage scripts
│   ├── emotion_example.py        # Emotion analysis example
│   └── genre_classification.py   # Genre classification example
├── configs/                      # Configuration files
│   ├── model_configs.py          # Model configurations
│   └── training_configs.py       # Training configurations
├── tests/                        # Test files
│   ├── test_data_split.py        # Data splitting tests
│   └── test_emotion_evaluation.py # Emotion evaluation tests
├── docs/                         # Documentation
│   └── Mel_Spectrogram_Technical_Guide.pdf
├── shell_scripts/                # Shell scripts
│   ├── run_training.sh           # Training execution script
│   ├── run_emotion.sh            # Emotion training script
│   ├── run_emotion_feedback.sh   # Emotion feedback script
│   ├── setup_environment.sh      # Environment setup script
│   ├── sync_to_remote.sh         # Remote sync script
│   └── sync_logs_from_remote.sh  # Log sync script
├── requirements.txt              # Python dependencies
├── README.md                     # Main documentation
└── PROJECT_STRUCTURE.md          # This file
```

## Key Improvements

### 1. **Modular Organization**
- **`src/`**: All source code organized by functionality
- **`scripts/`**: Executable scripts for common tasks
- **`configs/`**: Centralized configuration management
- **`examples/`**: Usage examples and demonstrations
- **`tests/`**: Test files for validation
- **`shell_scripts/`**: Shell scripts for automation

### 2. **Clear Separation of Concerns**
- **Models**: Neural network architectures
- **Data**: Data processing and feature extraction
- **Training**: Training and evaluation logic
- **Utils**: Helper functions and utilities

### 3. **Configuration Management**
- **`configs/model_configs.py`**: Model-specific configurations
- **`configs/training_configs.py`**: Training hyperparameters
- Centralized parameter management

### 4. **Improved Scripts**
- **`scripts/train.py`**: Main training script with argument parsing
- **`scripts/evaluate.py`**: Model evaluation script
- **`scripts/test_models.py`**: Model testing and validation
- **`scripts/extract_features.py`**: Feature extraction pipeline

### 5. **Better Documentation**
- Updated README with clear structure
- Example scripts for common use cases
- Configuration documentation

## Migration Guide

### Old Structure → New Structure

| Old Location | New Location | Purpose |
|-------------|-------------|---------|
| `pytorch/models.py` | `src/models/cnn_models.py` | CNN models |
| `pytorch/models_lrm.py` | `src/models/emotion_models.py` | Emotion models |
| `pytorch/losses.py` | `src/models/losses.py` | Loss functions |
| `pytorch/data_generator.py` | `src/data/data_generator.py` | Data loading |
| `utils/features.py` | `src/data/feature_extractor.py` | Feature extraction |
| `utils/data_generator.py` | `src/data/dataset_utils.py` | Dataset utilities |
| `pytorch/emotion_main.py` | `src/training/trainer.py` | Training loop |
| `pytorch/emotion_evaluate.py` | `src/training/evaluator.py` | Evaluation |
| `pytorch/emotion_evaluate_lrm.py` | `src/training/evaluator_lrm.py` | Emotion evaluation |
| `utils/utilities.py` | `src/utils/audio_utils.py` | Audio utilities |
| `plot_training_comparison.py` | `src/utils/visualization.py` | Visualization |
| `example_cnn6_emotion.py` | `examples/emotion_example.py` | Emotion example |
| `extract_emotion_features.py` | `scripts/extract_features.py` | Feature extraction script |

## Usage Examples

### Training a Model
```bash
# Train genre classification model
python scripts/train.py \
    --dataset_dir /path/to/gtzan \
    --workspace ./workspace \
    --model_type Transfer_Cnn14 \
    --learning_rate 1e-4 \
    --batch_size 32
```

### Evaluating a Model
```bash
# Evaluate trained model
python scripts/evaluate.py \
    --checkpoint_path ./workspace/checkpoints/model.pth \
    --model_type Transfer_Cnn14 \
    --dataset_dir /path/to/gtzan
```

### Testing Models
```bash
# Test all models
python scripts/test_models.py
```

### Extracting Features
```bash
# Extract audio features
python scripts/extract_features.py \
    --dataset_dir /path/to/gtzan \
    --workspace ./workspace
```

## Benefits of New Structure

1. **Maintainability**: Clear separation makes code easier to maintain
2. **Reusability**: Modular design allows components to be reused
3. **Testability**: Isolated components are easier to test
4. **Scalability**: Easy to add new models, data processors, or utilities
5. **Documentation**: Better organized documentation and examples
6. **Configuration**: Centralized configuration management
7. **Scripts**: Clear executable scripts for common tasks

## Next Steps

1. Update import statements in moved files to use new paths
2. Test all scripts to ensure they work with new structure
3. Update any hardcoded paths in configuration files
4. Add more comprehensive tests
5. Create additional example scripts for specific use cases 