#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model evaluation script for music genre classification and emotion regression.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.training.trainer import inference
from configs.training_configs import create_parser

def create_eval_parser():
    """Create argument parser for evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['Transfer_Cnn6', 'Transfer_Cnn14', 'FeatureAffectiveCnn6',
                               'FeatureEmotionRegression_Cnn6', 'FeatureEmotionRegression_Cnn6_NewAffective',
                               'FeatureEmotionRegression_Cnn6_LRM', 'EmotionRegression_Cnn6'],
                       help='Type of model to evaluate')
    
    # Dataset arguments
    parser.add_argument('--dataset_dir', type=str, default=None,
                       help='Path to GTZAN dataset directory (for genre classification)')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to emotion features HDF5 file (for emotion regression)')
    parser.add_argument('--workspace', type=str, default='./workspace',
                       help='Path to workspace directory')
    parser.add_argument('--holdout_fold', type=str, default='1',
                       help='Holdout fold for cross-validation')
    
    # Evaluation arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--forward_passes', type=int, default=2,
                       help='Number of forward passes for LRM models (default: 2)')
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    
    return parser

def main():
    """Main evaluation function."""
    parser = create_eval_parser()
    args = parser.parse_args()
    
    # Determine if this is emotion or genre evaluation based on model type
    emotion_models = ['FeatureEmotionRegression_Cnn6', 'FeatureEmotionRegression_Cnn6_NewAffective', 
                     'EmotionRegression_Cnn6', 'FeatureEmotionRegression_Cnn6_LRM']
    is_emotion_evaluation = args.model_type in emotion_models
    
    if is_emotion_evaluation:
        print("🎵 Emotion Regression Evaluation")
        print("=" * 50)
        print(f"Model: {args.model_type}")
        print(f"Checkpoint: {args.model_path}")
        print(f"Dataset: {args.dataset_path}")
        print(f"GPU: {args.gpu_id if args.cuda else 'CPU'}")
        print("=" * 50)
    else:
        print("🎵 GTZAN Music Genre Classification Evaluation")
        print("=" * 50)
        print(f"Model: {args.model_type}")
        print(f"Checkpoint: {args.model_path}")
        print(f"Dataset: {args.dataset_dir}")
        print(f"GPU: {args.gpu_id if args.cuda else 'CPU'}")
        print("=" * 50)
    
    # Validate checkpoint path
    if not Path(args.model_path).exists():
        print(f"❌ Checkpoint not found: {args.model_path}")
        sys.exit(1)
    
    # Set CUDA device
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        print(f"✅ Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    elif args.cuda and not torch.cuda.is_available():
        print("⚠️ CUDA requested but not available, using CPU")
        args.cuda = False
    else:
        print("🖥️ Using CPU")
    
    # Run evaluation using the trainer's inference function
    try:
        inference(args)
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main() 