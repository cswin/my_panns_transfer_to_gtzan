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

from src.utils.pytorch_utils import move_data_to_device, do_mixup
from src.utils.audio_utils import (create_folder, get_filename, create_logging, StatisticsContainer, Mixup)
from src.utils.config import sample_rate, mel_bins, fmin, fmax, window_size, hop_size, cnn14_config, cnn6_config
from src.training.losses import get_loss_func
from src.data.data_generator import EmoSoundscapesDataset, EmotionTrainSampler, EmotionValidateSampler, emotion_collate_fn, GtzanDataset, TrainSampler, EvaluateSampler, collate_fn
from src.models.cnn_models import FeatureEmotionRegression_Cnn14, EmotionRegression_Cnn14, FeatureEmotionRegression_Cnn6, EmotionRegression_Cnn6, FeatureEmotionRegression_Cnn6_NewAffective, Transfer_Cnn6, Transfer_Cnn14, FeatureAffectiveCnn6
from src.models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
from src.training.evaluator import EmotionEvaluator
from src.training.evaluator_lrm import LRMEmotionEvaluator


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
    
    # Set CUDA device and use only one GPU
    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = f'cuda:{args.gpu_id}'
        logging.info(f'Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}')
    else:
        device = 'cpu'
        logging.info('Using CPU')

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
    else:
        raise ValueError(f'Unknown model type: {args.model_type}')

    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    # Load pretrained weights
    if pretrained_checkpoint_path:
        logging.info(f'Loading pretrained model from {pretrained_checkpoint_path}')
        model.load_from_pretrain(pretrained_checkpoint_path)

    # Resume training
    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(resume_iteration))
        logging.info('Loading resume model from {}'.format(resume_checkpoint_path))
        resume_checkpoint = torch.load(resume_checkpoint_path, weights_only=False, map_location='cpu')
        model.load_state_dict(resume_checkpoint['model'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = resume_checkpoint['iteration']
    else:
        iteration = 0

    # Single GPU training (no DataParallel)
    print('Using single GPU: {}'.format(device))
    model = model.to(device)

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
                
                # Log segment-level metrics only if they exist (for segmented data)
                if 'segment_mean_mae' in statistics:
                    logging.info('Validate Segment Mean MAE: {:.4f}'.format(statistics['segment_mean_mae']))
                    logging.info('Validate Segment Mean Pearson: {:.4f}'.format(statistics['segment_mean_pearson']))
                elif 'audio_mean_mae' in statistics:
                    # For full-length audios, we have audio_ metrics instead of segment_ metrics
                    logging.info('Using full-length audio format (no segments)')
                else:
                    logging.info('No segment or audio metrics found in statistics')

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
                        'model': model.state_dict(), 
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
                'model': model.state_dict(), 
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
    batch_size = len(x)
    
    # Ensure batch size is even for mixup (drop last sample if odd)
    if batch_size % 2 == 1:
        # Remove the last sample to make batch size even
        x = x[:-1]
        valence_target = valence_target[:-1]
        arousal_target = arousal_target[:-1]
        batch_size = len(x)
    
    # Only apply mixup if we have at least 2 samples
    if batch_size >= 2:
        mixup_lambda = mixup_augmenter.get_lambda(batch_size)
        
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
    else:
        raise ValueError(f'Unknown model type: {args.model_type}')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    # Single GPU inference (no DataParallel)
    model = model.to(device)
    
    # Create data loader
    dataset = EmoSoundscapesDataset()
    validate_sampler = EmotionValidateSampler(hdf5_path=dataset_path, batch_size=batch_size)
    validate_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_sampler=validate_sampler, 
        collate_fn=emotion_collate_fn, num_workers=8, pin_memory=True)
    
    # Evaluate with CSV saving - use regular evaluator for full-length audios
    # LRM evaluator was designed for segment-based processing, but we now use full-length audios
    evaluator = EmotionEvaluator(model=model)
    print('Using standard evaluator for full-length audio processing')
    
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
        from src.utils.emotion_visualize import create_emotion_visualizations
        create_emotion_visualizations(output_dir)
        print(f"Visualizations saved in: {output_dir}")
    except ImportError as e:
        print(f"emotion_visualize module not found: {e}")
        print("Install matplotlib and seaborn to generate plots.")
        print("You can manually run: python src/utils/emotion_visualize.py <output_dir>")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        print("You can manually run: python src/utils/emotion_visualize.py <output_dir>")
    
    return statistics, output_dict


def train_genre(dataset_dir, workspace, holdout_fold, model_type, pretrained_checkpoint_path=None, 
                freeze_base=False, loss_type='clip_nll', augmentation='none', learning_rate=1e-4, 
                batch_size=32, resume_iteration=0, stop_iteration=10000, cuda=True, gpu_id=0):
    """Main training function for music genre classification."""
    
    # Set CUDA device and use only one GPU
    if cuda and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f'cuda:{gpu_id}'
        print(f'Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}')
    else:
        device = 'cpu'
        print('Using CPU')

    # Create directories
    checkpoints_dir = os.path.join(workspace, 'checkpoints', model_type)
    create_folder(checkpoints_dir)
    
    statistics_path = os.path.join(workspace, 'statistics', model_type, 'statistics.pickle')
    create_folder(os.path.dirname(statistics_path))
    
    logs_dir = os.path.join(workspace, 'logs', model_type)
    create_logging(logs_dir, 'w')
    
    logging.info(f'Training emotion regression model: {model_type}')
    logging.info(f'Dataset: {dataset_dir}')
    logging.info(f'Workspace: {workspace}')
    logging.info(f'Device: {device}')
    logging.info(f'Learning Rate: {learning_rate}')
    logging.info(f'Batch Size: {batch_size}')
    logging.info(f'GPU: {gpu_id}')
    
    # Model creation
    if model_type == 'FeatureEmotionRegression_Cnn14':
        config = cnn14_config
        model = FeatureEmotionRegression_Cnn14(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=freeze_base)
    elif model_type == 'EmotionRegression_Cnn14':
        config = cnn14_config
        model = EmotionRegression_Cnn14(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=freeze_base)
    elif model_type == 'FeatureEmotionRegression_Cnn6':
        config = cnn6_config
        model = FeatureEmotionRegression_Cnn6(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=freeze_base)
    elif model_type == 'EmotionRegression_Cnn6':
        config = cnn6_config
        model = EmotionRegression_Cnn6(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=freeze_base)
    elif model_type == 'FeatureEmotionRegression_Cnn6_LRM':
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
    elif model_type == 'FeatureEmotionRegression_Cnn6_NewAffective':
        config = cnn6_config
        model = FeatureEmotionRegression_Cnn6_NewAffective(
            sample_rate=sample_rate, 
            window_size=config['window_size'], 
            hop_size=config['hop_size'], 
            mel_bins=config['mel_bins'], 
            fmin=config['fmin'], 
            fmax=config['fmax'], 
            freeze_base=freeze_base)
    else:
        raise ValueError(f'Unknown model type: {model_type}')
    
    # Load pretrained weights
    if pretrained_checkpoint_path:
        logging.info(f'Loading pretrained model from {pretrained_checkpoint_path}')
        model.load_from_pretrain(pretrained_checkpoint_path)
    
    # Move model to device (single GPU)
    model = model.to(device)
    
    # Load resume checkpoint if specified
    if resume_iteration > 0:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(resume_iteration))
        logging.info(f'Loading resume checkpoint from {resume_checkpoint_path}')
        resume_checkpoint = torch.load(resume_checkpoint_path, weights_only=False)
        model.load_state_dict(resume_checkpoint['model'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = resume_checkpoint['iteration']
    else:
        iteration = 0

    # Single GPU training (no DataParallel)
    print('Using single GPU: {}'.format(device))

    # Dataset and data loaders
    dataset = GtzanDataset()
    
    # Create feature file path
    feature_file = os.path.join(workspace, 'features', 'features.h5')
    
    # Data samplers
    train_sampler = TrainSampler(
        hdf5_path=feature_file,
        holdout_fold=holdout_fold,
        batch_size=batch_size
    )
    
    validate_sampler = EvaluateSampler(
        hdf5_path=feature_file,
        holdout_fold=holdout_fold,
        batch_size=batch_size
    )
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )
    
    validate_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=validate_sampler,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Loss function
    if loss_type == 'clip_nll':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Unknown loss type: {loss_type}')
    
    # Training loop
    iteration = 0
    for batch_data_dict in train_loader:
        
        # Move data to device
        batch_feature = move_data_to_device(batch_data_dict['feature'], device)
        batch_target = move_data_to_device(batch_data_dict['target'], device)
        
        # Forward pass (model will add channel dimension internally)
        model.train()
        batch_output_dict = model(batch_feature)
        batch_output = batch_output_dict['clipwise_output']
        
        # Calculate loss
        loss = criterion(batch_output, batch_target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if iteration % 100 == 0:
            print(f'Iteration: {iteration}, loss: {loss.item():.3f}')
        
        # Evaluate periodically
        if iteration % 500 == 0 and iteration > 0:
            model.eval()
            with torch.no_grad():
                total_loss = 0
                correct = 0
                total = 0
                num_batches = 0
                
                for val_batch in validate_loader:
                    val_feature = move_data_to_device(val_batch['feature'], device)
                    val_target = move_data_to_device(val_batch['target'], device)
                    
                    val_output_dict = model(val_feature)
                    val_output = val_output_dict['clipwise_output']
                    
                    val_loss = criterion(val_output, val_target)
                    total_loss += val_loss.item()
                    
                    _, predicted = torch.max(val_output.data, 1)
                    _, target_labels = torch.max(val_target.data, 1)
                    total += target_labels.size(0)
                    correct += (predicted == target_labels).sum().item()
                    num_batches += 1
                
                accuracy = 100 * correct / total
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                print(f'Validation - Loss: {avg_loss:.3f}, Accuracy: {accuracy:.2f}%')
        
        iteration += 1
        
        # Stop training
        if iteration >= stop_iteration:
            break
    
    print(f'Training completed! Final iteration: {iteration}')
    return model


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
                              choices=['FeatureEmotionRegression_Cnn14', 'EmotionRegression_Cnn14', 'FeatureEmotionRegression_Cnn6', 'EmotionRegression_Cnn6', 'FeatureEmotionRegression_Cnn6_LRM', 'FeatureEmotionRegression_Cnn6_NewAffective'])
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
    parser_train.add_argument('--gpu_id', type=int, default=0,
                              help='GPU ID to use')

    # Inference arguments
    parser_inference = subparsers.add_parser('inference')
    parser_inference.add_argument('--model_path', type=str, required=True,
                                  help='Path to trained model checkpoint')
    parser_inference.add_argument('--dataset_path', type=str, required=True,
                                  help='Path to emotion dataset HDF5 file')
    parser_inference.add_argument('--model_type', type=str, required=True,
                                  choices=['FeatureEmotionRegression_Cnn14', 'EmotionRegression_Cnn14', 'FeatureEmotionRegression_Cnn6', 'EmotionRegression_Cnn6', 'FeatureEmotionRegression_Cnn6_LRM', 'FeatureEmotionRegression_Cnn6_NewAffective'])
    parser_inference.add_argument('--batch_size', type=int, default=32)
    parser_inference.add_argument('--cuda', action='store_true')
    parser_inference.add_argument('--forward_passes', type=int, default=2,
                                  help='Number of forward passes for feedback models')

    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'train_genre':
        train_genre(args.dataset_path, args.workspace, args.holdout_fold, args.model_type, args.pretrained_checkpoint_path, args.freeze_base, args.loss_type, args.augmentation, args.learning_rate, args.batch_size, args.resume_iteration, args.stop_iteration, args.cuda, args.gpu_id)
    else:
        raise ValueError('Invalid mode!') 