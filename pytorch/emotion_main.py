#!/usr/bin/env python3
"""
Main script for training emotion regression models on Emo-Soundscapes dataset.

This script trains PANNs models to predict valence and arousal from audio clips.
"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import sample_rate, mel_bins, fmin, fmax, window_size, hop_size, cnn14_config, cnn6_config
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer, Mixup)
from data_generator import EmoSoundscapesDataset, EmotionTrainSampler, EmotionValidateSampler, emotion_collate_fn
from models import FeatureEmotionRegression_Cnn14, EmotionRegression_Cnn14, FeatureEmotionRegression_Cnn6, EmotionRegression_Cnn6, FeatureEmotionRegression_Cnn6_NewAffective
from models_lrm import FeatureEmotionRegression_Cnn6_LRM
from emotion_evaluate import EmotionEvaluator
from emotion_evaluate_lrm import LRMEmotionEvaluator


def train(args):
    """Main training function for emotion regression."""

    # Arguments & parameters
    dataset_path = args.dataset_path
    workspace = args.workspace
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    resume_iteration = args.resume_iteration
    stop_iteration = args.stop_iteration
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename

    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False

    # Create directories
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        model_type, 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation),
        'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename, 
        model_type, 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base), 
        'statistics.pickle')
    create_folder(os.path.dirname(statistics_path))
    
    logs_dir = os.path.join(workspace, 'logs', filename, 
        model_type, 'pretrain={}'.format(pretrain), 
        'loss_type={}'.format(loss_type), 'augmentation={}'.format(augmentation), 
        'batch_size={}'.format(batch_size), 'freeze_base={}'.format(freeze_base))
    create_logging(logs_dir, 'w')
    logging.info(args)

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
    
    # Model
    if args.model_type == 'FeatureEmotionRegression_Cnn14':
        config = cnn14_config
        model = FeatureEmotionRegression_Cnn14(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=freeze_base)
    elif args.model_type == 'EmotionRegression_Cnn14':
        config = cnn14_config
        model = EmotionRegression_Cnn14(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=freeze_base)
    elif args.model_type == 'FeatureEmotionRegression_Cnn6':
        config = cnn6_config
        model = FeatureEmotionRegression_Cnn6(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=freeze_base)
    elif args.model_type == 'EmotionRegression_Cnn6':
        config = cnn6_config
        model = EmotionRegression_Cnn6(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=freeze_base)
    elif args.model_type == 'FeatureEmotionRegression_Cnn6_NewAffective':
        config = cnn6_config
        model = FeatureEmotionRegression_Cnn6_NewAffective(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=freeze_base)
    elif args.model_type == 'FeatureEmotionRegression_Cnn6_LRM':
        config = cnn6_config
        model = FeatureEmotionRegression_Cnn6_LRM(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=freeze_base,
            forward_passes=getattr(args, 'forward_passes', 2))
    else:
        raise ValueError(f'Unknown model type: {args.model_type}')

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    # Load pretrained weights
    if pretrain:
        logging.info('Loading pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    # Resume training
    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(resume_iteration))
        logging.info('Loading resume model from {}'.format(resume_checkpoint_path))
        resume_checkpoint = torch.load(resume_checkpoint_path, weights_only=False)
        model.load_state_dict(resume_checkpoint['model'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = resume_checkpoint['iteration']
    else:
        iteration = 0

    # Parallel training
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    # Dataset
    dataset = EmoSoundscapesDataset()

    # Data samplers
    train_sampler = EmotionTrainSampler(
        hdf5_path=dataset_path, 
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size)

    validate_sampler = EmotionValidateSampler(
        hdf5_path=dataset_path, 
        batch_size=batch_size)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_sampler=train_sampler, 
        collate_fn=emotion_collate_fn, 
        num_workers=8, 
        pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_sampler=validate_sampler, 
        collate_fn=emotion_collate_fn, 
        num_workers=8, 
        pin_memory=True)

    if 'cuda' in device:
        model.to(device)

    # Optimizer - use same settings for both models for fair comparison
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                          betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
    
    # Stronger regularization only for LRM models (but same learning rate)
    if 'LRM' in args.model_type:
        # Only add weight decay for LRM models to handle additional feedback parameters
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                              betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=True)
        logging.info('Using stronger regularization (weight_decay=0.01) for LRM model')
    
    # Best model tracking for final evaluation (but continue training)
    best_val_pearson = -1.0
    best_model_path = None
    
    # Learning rate scheduler for LRM models
    if 'LRM' in args.model_type:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                                        patience=5, min_lr=1e-6)
        logging.info('Using learning rate scheduler for LRM model')
    else:
        scheduler = None

    # Mixup augmentation
    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)
     
    # Evaluator - use LRM evaluator for LRM models
    if 'LRM' in args.model_type:
        evaluator = LRMEmotionEvaluator(model=model)
        logging.info('Using LRM evaluator for segment-based feedback processing')
    else:
        evaluator = EmotionEvaluator(model=model)
    
    train_bgn_time = time.time()
    
    # Training loop - cycle through data loader indefinitely
    import itertools
    train_loader_cycle = itertools.cycle(train_loader)
    
    for batch_data_dict in train_loader_cycle:
        
        # Evaluate periodically
        if iteration % 200 == 0 and iteration > 0:
            if resume_iteration > 0 and iteration == resume_iteration:
                pass
            else:
                logging.info('------------------------------------')
                logging.info('Iteration: {}'.format(iteration))

                train_fin_time = time.time()

                # Evaluate on validation set
                statistics = evaluator.evaluate(validate_loader)
                
                # Log main metrics (use audio-level metrics for primary evaluation)
                logging.info('Validate Audio Mean MAE: {:.4f}'.format(statistics['audio_mean_mae']))
                logging.info('Validate Audio Mean RMSE: {:.4f}'.format(statistics['audio_mean_rmse']))
                logging.info('Validate Audio Mean Pearson: {:.4f}'.format(statistics['audio_mean_pearson']))
                logging.info('Validate Audio Valence MAE: {:.4f}, Arousal MAE: {:.4f}'.format(
                    statistics['audio_valence_mae'], statistics['audio_arousal_mae']))
                # Log separate valence and arousal Pearson correlations
                logging.info('Validate Audio Valence Pearson: {:.4f}, Arousal Pearson: {:.4f}'.format(
                    statistics['audio_valence_pearson'], statistics['audio_arousal_pearson']))
                
                # Also log segment-level for comparison
                logging.info('Validate Segment Mean MAE: {:.4f}'.format(statistics['segment_mean_mae']))
                logging.info('Validate Segment Mean Pearson: {:.4f}'.format(statistics['segment_mean_pearson']))

                # Append statistics
                statistics_container.append(iteration, statistics, data_type='validate')
                statistics_container.dump()

                # Best model tracking for all models (good practice for anti-overfitting)
                current_val_pearson = statistics['audio_mean_pearson']
                
                # Track best model for final evaluation (but continue training)
                if current_val_pearson > best_val_pearson:
                    best_val_pearson = current_val_pearson
                    # Save best model
                    best_model_path = os.path.join(checkpoints_dir, 'best_model.pth')
                    checkpoint = {
                        'iteration': iteration, 
                        'model': model.module.state_dict(), 
                        'optimizer': optimizer.state_dict(),
                        'best_val_pearson': best_val_pearson}
                    torch.save(checkpoint, best_model_path)
                    logging.info('New best model saved with validation Pearson: {:.4f}'.format(best_val_pearson))

                # Learning rate scheduling for LRM models only
                if 'LRM' in args.model_type:
                    # Update learning rate scheduler
                    if scheduler is not None:
                        scheduler.step(current_val_pearson)

                train_time = train_fin_time - train_bgn_time
                validate_time = time.time() - train_fin_time

                logging.info('Train time: {:.3f} s, validate time: {:.3f} s'.format(
                    train_time, validate_time))

                train_bgn_time = time.time()
        
        # Save model checkpoint
        if iteration % 1000 == 0 and iteration > 0:
            checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(iteration))
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict(), 
                'optimizer': optimizer.state_dict()}

            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        # Stop training
        if iteration == stop_iteration:
            break
            
        # Move data to device
        batch_feature = move_data_to_device(batch_data_dict['feature'], device)
        batch_valence_target = move_data_to_device(batch_data_dict['valence'], device)
        batch_arousal_target = move_data_to_device(batch_data_dict['arousal'], device)

        # Add channel dimension for feature input
        if len(batch_feature.shape) == 3:  # (batch_size, time_steps, mel_bins)
            batch_feature = batch_feature.unsqueeze(1)  # (batch_size, 1, time_steps, mel_bins)

        # Mixup augmentation
        if 'mixup' in augmentation:
            batch_feature, batch_valence_target, batch_arousal_target = do_mixup_emotion(
                batch_feature, batch_valence_target, batch_arousal_target, mixup_augmenter)

        # Forward pass
        model.train()
        batch_output_dict = model(batch_feature, None)

        # Calculate loss
        target_dict = {
            'valence': batch_valence_target,
            'arousal': batch_arousal_target
        }
        loss = loss_func(batch_output_dict, target_dict)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss occasionally
        if iteration % 100 == 0:
            print('Iteration: {}, loss: {:.3f}'.format(iteration, loss.item()))

        iteration += 1


def do_mixup_emotion(x, valence_target, arousal_target, mixup_augmenter):
    """Apply mixup augmentation for emotion regression."""
    mixup_lambda = mixup_augmenter.get_lambda(batch_size=len(x))
    
    # Apply mixup to features
    x = do_mixup(x, mixup_lambda)
    
    # Apply mixup to targets
    valence_target = do_mixup(valence_target, mixup_lambda)
    arousal_target = do_mixup(arousal_target, mixup_lambda)
    
    return x, valence_target, arousal_target


def inference(args):
    """Inference mode - evaluate trained model."""
    
    # Load model
    model_path = args.model_path
    dataset_path = args.dataset_path
    batch_size = args.batch_size
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    
    # Create model
    if args.model_type == 'FeatureEmotionRegression_Cnn14':
        config = cnn14_config
        model = FeatureEmotionRegression_Cnn14(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=True)
    elif args.model_type == 'EmotionRegression_Cnn14':
        config = cnn14_config
        model = EmotionRegression_Cnn14(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=True)
    elif args.model_type == 'FeatureEmotionRegression_Cnn6':
        config = cnn6_config
        model = FeatureEmotionRegression_Cnn6(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=True)
    elif args.model_type == 'EmotionRegression_Cnn6':
        config = cnn6_config
        model = EmotionRegression_Cnn6(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=True)
    elif args.model_type == 'FeatureEmotionRegression_Cnn6_NewAffective':
        config = cnn6_config
        model = FeatureEmotionRegression_Cnn6_NewAffective(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=True)
    elif args.model_type == 'FeatureEmotionRegression_Cnn6_LRM':
        config = cnn6_config
        model = FeatureEmotionRegression_Cnn6_LRM(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=True,
            forward_passes=getattr(args, 'forward_passes', 2))
    else:
        raise ValueError(f'Unknown model type: {args.model_type}')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    
    model = torch.nn.DataParallel(model)
    if 'cuda' in device:
        model.to(device)
    
    # Create data loader
    dataset = EmoSoundscapesDataset()
    validate_sampler = EmotionValidateSampler(hdf5_path=dataset_path, batch_size=batch_size)
    validate_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_sampler=validate_sampler, 
        collate_fn=emotion_collate_fn, num_workers=8, pin_memory=True)
    
    # Evaluate with CSV saving - use LRM evaluator for LRM models
    if 'LRM' in args.model_type:
        evaluator = LRMEmotionEvaluator(model=model)
        print('Using LRM evaluator for segment-based feedback processing')
    else:
        evaluator = EmotionEvaluator(model=model)
    
    # Create output directory for predictions based on model path
    # Extract workspace from model path to save predictions in the same workspace
    model_dir = os.path.dirname(model_path)
    if 'workspaces' in model_dir:
        # Find the workspace directory (e.g., workspaces/emotion_feedback_stable)
        workspace_parts = model_dir.split(os.sep)
        workspace_idx = workspace_parts.index('workspaces')
        if workspace_idx + 1 < len(workspace_parts):
            workspace_base = os.path.join(*workspace_parts[:workspace_idx+2])
        else:
            # Fallback to model type-based logic
            workspace_base = 'workspaces/emotion_feedback' if 'LRM' in args.model_type else 'workspaces/emotion_regression'
    else:
        # Fallback to model type-based logic
        workspace_base = 'workspaces/emotion_feedback' if 'LRM' in args.model_type else 'workspaces/emotion_regression'
    
    output_dir = os.path.join(workspace_base, 'predictions')
    
    # Save predictions and get statistics
    statistics, output_dict = evaluator.evaluate(validate_loader, save_predictions=True, output_dir=output_dir)
    
    # Print results
    evaluator.print_evaluation(statistics)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        # Fix import path - add parent directory to sys.path to find emotion_visualize
        import sys
        parent_dir = os.path.join(os.path.dirname(__file__), '..')
        sys.path.insert(0, os.path.abspath(parent_dir))
        
        from emotion_visualize import create_emotion_visualizations
        create_emotion_visualizations(output_dir)
        print(f"Visualizations saved in: {output_dir}")
    except ImportError as e:
        print(f"emotion_visualize module not found: {e}")
        print("Install matplotlib and seaborn to generate plots.")
        print("You can manually run: python generate_emotion_plots.py <output_dir>")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        print("You can manually run: python generate_emotion_plots.py <output_dir>")
    
    return statistics, output_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train emotion regression models')
    subparsers = parser.add_subparsers(dest='mode')

    # Training arguments
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_path', type=str, required=True,
                              help='Path to emotion dataset HDF5 file')
    parser_train.add_argument('--workspace', type=str, required=True, 
                              help='Directory of workspace')
    parser_train.add_argument('--model_type', type=str, required=True,
                              choices=['FeatureEmotionRegression_Cnn14', 'EmotionRegression_Cnn14', 'FeatureEmotionRegression_Cnn6', 'EmotionRegression_Cnn6', 'FeatureEmotionRegression_Cnn6_NewAffective', 'FeatureEmotionRegression_Cnn6_LRM'])
    parser_train.add_argument('--pretrained_checkpoint_path', type=str,
                              help='Path to pretrained model checkpoint')
    parser_train.add_argument('--freeze_base', action='store_true', 
                              help='Freeze pretrained layers')
    parser_train.add_argument('--loss_type', type=str, default='mse',
                              choices=['mse', 'mae', 'smooth_l1'])
    parser_train.add_argument('--augmentation', type=str, default='none',
                              choices=['none', 'mixup'])
    parser_train.add_argument('--learning_rate', type=float, default=1e-4)
    parser_train.add_argument('--batch_size', type=int, default=32)
    parser_train.add_argument('--resume_iteration', type=int, default=0)
    parser_train.add_argument('--stop_iteration', type=int, default=10000)
    parser_train.add_argument('--cuda', action='store_true')
    parser_train.add_argument('--filename', type=str, default='emotion_main')
    parser_train.add_argument('--forward_passes', type=int, default=2,
                              help='Number of forward passes for feedback models')

    # Inference arguments
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--model_path', type=str, required=True,
                                  help='Path to trained model checkpoint')
    parser_inference.add_argument('--dataset_path', type=str, required=True,
                                  help='Path to emotion dataset HDF5 file')
    parser_inference.add_argument('--model_type', type=str, required=True,
                                  choices=['FeatureEmotionRegression_Cnn14', 'EmotionRegression_Cnn14', 'FeatureEmotionRegression_Cnn6', 'EmotionRegression_Cnn6', 'FeatureEmotionRegression_Cnn6_NewAffective', 'FeatureEmotionRegression_Cnn6_LRM'])
    parser_inference.add_argument('--batch_size', type=int, default=32)
    parser_inference.add_argument('--cuda', action='store_true')
    parser_inference.add_argument('--forward_passes', type=int, default=2,
                                  help='Number of forward passes for feedback models')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    else:
        raise ValueError('Invalid mode!') 