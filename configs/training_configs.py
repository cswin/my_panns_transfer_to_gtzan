"""
Training configurations and hyperparameters for model training.
"""

import argparse
from pathlib import Path

# Default training parameters
DEFAULT_TRAINING_CONFIG = {
    # Dataset
    'dataset_dir': '/path/to/gtzan/dataset',
    'workspace': './workspace',
    'holdout_fold': 1,
    
    # Model
    'model_type': 'Transfer_Cnn14',
    'pretrained_checkpoint_path': None,
    'freeze_base': False,
    
    # Training
    'learning_rate': 1e-4,
    'batch_size': 32,
    'resume_iteration': 0,
    'stop_iteration': 10000,
    'loss_type': 'clip_nll',
    'augmentation': 'none',
    
    # Hardware
    'cuda': True,
    'gpu_id': 0,
    
    # Logging
    'filename': 'main',
    'mode': 'train',
}

# Model-specific configurations
MODEL_TRAINING_CONFIGS = {
    'Transfer_Cnn6': {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'stop_iteration': 8000,
    },
    'Transfer_Cnn14': {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'stop_iteration': 10000,
    },
    'FeatureAffectiveCnn6': {
        'learning_rate': 5e-5,
        'batch_size': 16,
        'stop_iteration': 12000,
    },
    'FeatureEmotionRegression_Cnn6': {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'stop_iteration': 15000,
        'loss_type': 'emotion_loss',
    },
    'FeatureEmotionRegression_Cnn6_NewAffective': {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'stop_iteration': 15000,
        'loss_type': 'emotion_loss',
    },
    'FeatureEmotionRegression_Cnn6_LRM': {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'stop_iteration': 15000,
        'loss_type': 'emotion_loss',
    },
    'EmotionRegression_Cnn6': {
        'learning_rate': 1e-4,
        'batch_size': 32,
        'stop_iteration': 15000,
        'loss_type': 'emotion_loss',
    },
}

def get_training_config(model_type=None, **kwargs):
    """Get training configuration with optional model-specific overrides."""
    config = DEFAULT_TRAINING_CONFIG.copy()
    
    if model_type and model_type in MODEL_TRAINING_CONFIGS:
        config.update(MODEL_TRAINING_CONFIGS[model_type])
    
    # Override with any provided kwargs
    config.update(kwargs)
    
    return config

def create_parser():
    """Create argument parser for training script."""
    parser = argparse.ArgumentParser(description='Train music genre classification and emotion regression models')
    
    # Dataset arguments
    parser.add_argument('--dataset_dir', type=str, default=None,
                       help='Path to GTZAN dataset directory (for genre classification)')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to emotion features HDF5 file (for emotion regression)')
    parser.add_argument('--workspace', type=str, default='./workspace',
                       help='Path to workspace directory')
    parser.add_argument('--holdout_fold', type=str, default='1',
                       help='Holdout fold for cross-validation')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='Transfer_Cnn14',
                       choices=['Transfer_Cnn6', 'Transfer_Cnn14', 'FeatureAffectiveCnn6',
                               'FeatureEmotionRegression_Cnn6', 'FeatureEmotionRegression_Cnn6_NewAffective',
                               'FeatureEmotionRegression_Cnn6_LRM', 'EmotionRegression_Cnn6'],
                       help='Type of model to train')
    parser.add_argument('--pretrained_checkpoint_path', type=str, default=None,
                       help='Path to pretrained model checkpoint')
    parser.add_argument('--freeze_base', action='store_true',
                       help='Freeze base model layers')
    
    # Training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--resume_iteration', type=int, default=0,
                       help='Resume training from iteration')
    parser.add_argument('--stop_iteration', type=int, default=10000,
                       help='Stop training at iteration')
    parser.add_argument('--loss_type', type=str, default='clip_nll',
                       choices=['clip_nll', 'emotion_loss', 'mse'],
                       help='Loss function type')
    parser.add_argument('--augmentation', type=str, default='none',
                       choices=['none', 'mixup', 'specaugment'],
                       help='Data augmentation method')
    parser.add_argument('--forward_passes', type=int, default=2,
                       help='Number of forward passes for LRM models (default: 2)')
    
    # Hardware arguments
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    
    # Logging arguments
    parser.add_argument('--filename', type=str, default='main',
                       help='Filename for logging')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate'],
                       help='Mode: train or evaluate')
    
    return parser

def validate_config(config):
    """Validate training configuration."""
    # Determine if this is emotion or genre training based on model type
    emotion_models = ['FeatureEmotionRegression_Cnn6', 'FeatureEmotionRegression_Cnn6_NewAffective', 
                     'FeatureEmotionRegression_Cnn6_LRM', 'EmotionRegression_Cnn6']
    is_emotion_training = config['model_type'] in emotion_models
    
    if is_emotion_training:
        # Emotion training validation
        if not config['dataset_path']:
            raise ValueError("dataset_path is required for emotion training")
        if not Path(config['dataset_path']).exists():
            raise ValueError(f"Emotion dataset file does not exist: {config['dataset_path']}")
    else:
        # Genre training validation
        if not config['dataset_dir']:
            raise ValueError("dataset_dir is required for genre classification training")
        if not Path(config['dataset_dir']).exists():
            raise ValueError(f"Dataset directory does not exist: {config['dataset_dir']}")
    
    # Check model type
    valid_models = ['Transfer_Cnn6', 'Transfer_Cnn14', 'FeatureAffectiveCnn6',
                   'FeatureEmotionRegression_Cnn6', 'FeatureEmotionRegression_Cnn6_NewAffective',
                   'FeatureEmotionRegression_Cnn6_LRM', 'EmotionRegression_Cnn6']
    if config['model_type'] not in valid_models:
        raise ValueError(f"Invalid model type: {config['model_type']}")
    
    # Check pretrained checkpoint
    if config['pretrained_checkpoint_path'] and not Path(config['pretrained_checkpoint_path']).exists():
        raise ValueError(f"Pretrained checkpoint does not exist: {config['pretrained_checkpoint_path']}")
    
    return config 