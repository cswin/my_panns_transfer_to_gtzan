#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model evaluation script for music genre classification and emotion regression.
"""

import os
import sys
import argparse
import torch
import glob
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.training.trainer import inference
from configs.training_configs import create_parser

def find_best_model(workspace, model_type, batch_size=None):
    """Find the best model checkpoint in the workspace."""
    # Look for best_model.pth in the workspace
    best_model_pattern = os.path.join(workspace, "checkpoints", "**", model_type, "**", "best_model.pth")
    best_models = glob.glob(best_model_pattern, recursive=True)
    
    if best_models:
        # If batch_size is specified, try to find a model with matching batch_size
        if batch_size is not None:
            matching_models = []
            for model_path in best_models:
                # Check if the path contains the batch_size
                if f"batch_size={batch_size}" in model_path:
                    matching_models.append(model_path)
            
            if matching_models:
                # Return the first matching model
                return matching_models[0]
            else:
                print(f"⚠️ No best model found with batch_size={batch_size}, using first available")
        
        # Return the first best model found (fallback)
        return best_models[0]
    
    # Fallback: look for latest iteration checkpoint
    latest_pattern = os.path.join(workspace, "checkpoints", "**", model_type, "**", "*_iterations.pth")
    latest_models = glob.glob(latest_pattern, recursive=True)
    
    if latest_models:
        # Sort by iteration number and return the latest
        def extract_iteration(path):
            filename = os.path.basename(path)
            if '_iterations.pth' in filename:
                try:
                    return int(filename.split('_')[0])
                except:
                    return 0
            return 0
        
        latest_models.sort(key=extract_iteration, reverse=True)
        return latest_models[0]
    
    return None

def create_eval_parser():
    """Create argument parser for evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    
    # Model arguments
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to trained model checkpoint (if not provided, will auto-find best model)')
    parser.add_argument('--use_best_model', action='store_true', default=False,
                       help='Force use of best model (overrides --model_path if specified)')
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
    
    # Auto-find best model if not provided or if --use_best_model is specified
    if args.use_best_model or args.model_path is None:
        print("🔍 Auto-searching for best model...")
        best_model_path = find_best_model(args.workspace, args.model_type, args.batch_size)
        
        if best_model_path:
            args.model_path = best_model_path
            if args.use_best_model:
                print(f"✅ Using best model: {args.model_path}")
            else:
                print(f"✅ Auto-selected best model: {args.model_path}")
        else:
            print(f"❌ No model found in workspace: {args.workspace}")
            print("Available options:")
            print("  1. Specify --model_path explicitly")
            print("  2. Check if workspace contains trained models")
            print("  3. Run training first to generate models")
            sys.exit(1)
    
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