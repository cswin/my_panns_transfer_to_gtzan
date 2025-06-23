#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for music genre classification and emotion regression with PANNs transfer learning.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.training.trainer import train_genre, train
from src.training.config import sample_rate, cnn6_config, cnn14_config
from src.models.cnn_models import Transfer_Cnn6, Transfer_Cnn14, FeatureAffectiveCnn6, EmotionRegression_Cnn6
from src.models.emotion_models import FeatureEmotionRegression_Cnn6
from configs.training_configs import create_parser, validate_config, get_training_config
from configs.model_configs import get_model_config

def main():
    """Main training function."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Validate configuration
    try:
        config = validate_config(config)
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    # Determine if this is emotion or genre training based on model type
    emotion_models = ['FeatureEmotionRegression_Cnn6', 'FeatureEmotionRegression_Cnn6_NewAffective', 
                     'EmotionRegression_Cnn6', 'FeatureEmotionRegression_Cnn6_LRM']
    is_emotion_training = config['model_type'] in emotion_models
    
    if is_emotion_training:
        print("🎵 Emotion Regression Training")
        print("=" * 50)
        print(f"Model: {config['model_type']}")
        print(f"Dataset: {config.get('dataset_path', 'emotion_features.h5')}")
        print(f"Workspace: {config['workspace']}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"GPU: {config['gpu_id'] if config['cuda'] else 'CPU'}")
        print("=" * 50)
    else:
        print("🎵 GTZAN Music Genre Classification Training")
        print("=" * 50)
        print(f"Model: {config['model_type']}")
        print(f"Dataset: {config['dataset_dir']}")
        print(f"Workspace: {config['workspace']}")
        print(f"Learning Rate: {config['learning_rate']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"GPU: {config['gpu_id'] if config['cuda'] else 'CPU'}")
        print("=" * 50)
    
    # Set CUDA device
    if config['cuda'] and torch.cuda.is_available():
        torch.cuda.set_device(config['gpu_id'])
        print(f"✅ Using GPU {config['gpu_id']}: {torch.cuda.get_device_name(config['gpu_id'])}")
    elif config['cuda'] and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available, using CPU")
        config['cuda'] = False
    else:
        print("🖥️ Using CPU")
    
    # Create workspace directories
    workspace = Path(config['workspace'])
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / 'checkpoints').mkdir(exist_ok=True)
    (workspace / 'logs').mkdir(exist_ok=True)
    (workspace / 'statistics').mkdir(exist_ok=True)
    
    # Start training
    try:
        if is_emotion_training:
            # Emotion training
            train(args)
        else:
            # Genre training
            train_genre(
                dataset_dir=config['dataset_dir'],
                workspace=config['workspace'],
                holdout_fold=config['holdout_fold'],
                model_type=config['model_type'],
                pretrained_checkpoint_path=config['pretrained_checkpoint_path'],
                freeze_base=config['freeze_base'],
                loss_type=config['loss_type'],
                augmentation=config['augmentation'],
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size'],
                resume_iteration=config['resume_iteration'],
                stop_iteration=config['stop_iteration'],
                cuda=config['cuda']
            )
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 