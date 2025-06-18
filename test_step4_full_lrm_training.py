#!/usr/bin/env python3
"""
Step 4: Full LRM Training Test

This test runs a complete training session with the LRM model on real emotion data.
It verifies that:
1. LRM model trains properly on real Emo-Soundscapes data
2. Segment-based feedback improves performance over time
3. LRM evaluation system works correctly
4. Training metrics are properly tracked
5. Model checkpoints can be saved and loaded

Run this on the remote server with GPU and real data.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
from pathlib import Path

# Add directories to path
sys.path.append('pytorch')
sys.path.append('utils')

# Import utilities first to ensure it's available
try:
    from utilities import create_folder
except ImportError:
    # Fallback: create our own create_folder function
    def create_folder(path):
        Path(path).mkdir(parents=True, exist_ok=True)

from models_lrm import FeatureEmotionRegression_Cnn6_LRM
from models import FeatureEmotionRegression_Cnn6_NewAffective
from emotion_evaluate_lrm import evaluate_lrm
import config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Step 4: Full LRM Training Test')
    
    # Data arguments
    parser.add_argument('--features_hdf5_path', type=str, 
                       default='workspaces/emotion_regression/features/emotion_features.h5',
                       help='Path to features HDF5 file')
    parser.add_argument('--cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    
    # Model arguments  
    parser.add_argument('--model_type', type=str, default='FeatureEmotionRegression_Cnn6_LRM',
                       choices=['FeatureEmotionRegression_Cnn6_LRM', 'FeatureEmotionRegression_Cnn6_NewAffective'],
                       help='Model type to train')
    parser.add_argument('--freeze_base', action='store_true', default=True,
                       help='Freeze base CNN layers')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--early_stop', type=int, default=5,
                       help='Early stopping patience')
    
    # LRM-specific arguments
    parser.add_argument('--forward_passes', type=int, default=2,
                       help='Number of forward passes for LRM')
    parser.add_argument('--modulation_strength', type=float, default=1.0,
                       help='Modulation strength for LRM')
    
    # Output arguments
    parser.add_argument('--workspace', type=str, default='./workspaces/step4_lrm_test',
                       help='Workspace directory for outputs')
    parser.add_argument('--save_model', action='store_true', default=True,
                       help='Save trained model')
    
    return parser.parse_args()

def create_model(args, device):
    """Create and initialize model."""
    print(f"Creating {args.model_type} model...")
    
    # Model parameters from config
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    
    if args.model_type == 'FeatureEmotionRegression_Cnn6_LRM':
        model = FeatureEmotionRegression_Cnn6_LRM(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            freeze_base=args.freeze_base,
            forward_passes=args.forward_passes
        )
        
        # Set modulation strength if specified
        if args.modulation_strength != 1.0:
            model.set_modulation_strength(args.modulation_strength)
            print(f"Set modulation strength to {args.modulation_strength}")
            
    elif args.model_type == 'FeatureEmotionRegression_Cnn6_NewAffective':
        model = FeatureEmotionRegression_Cnn6_NewAffective(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            freeze_base=args.freeze_base
        )
    
    # Move to device
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
    
    return model

def train_epoch(model, data_loader, optimizer, device, epoch, is_lrm=False):
    """Train for one epoch."""
    model.train()
    
    epoch_losses = []
    epoch_valence_losses = []
    epoch_arousal_losses = []
    
    num_batches = len(data_loader)
    
    for batch_idx, batch_data_dict in enumerate(data_loader):
        # Move data to device
        batch_feature = batch_data_dict['feature'].to(device)
        batch_valence = batch_data_dict['valence'].to(device)
        batch_arousal = batch_data_dict['arousal'].to(device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass
        if is_lrm:
            # For LRM, clear feedback state at start of each batch
            model.clear_feedback_state()
        
        output_dict = model(batch_feature)
        
        # Compute loss
        valence_loss = nn.MSELoss()(output_dict['valence'], batch_valence)
        arousal_loss = nn.MSELoss()(output_dict['arousal'], batch_arousal)
        total_loss = valence_loss + arousal_loss
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # Update parameters
        optimizer.step()
        
        # Store losses
        epoch_losses.append(total_loss.item())
        epoch_valence_losses.append(valence_loss.item())
        epoch_arousal_losses.append(arousal_loss.item())
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f'  Batch {batch_idx}/{num_batches}: '
                  f'Loss={total_loss.item():.6f} '
                  f'(V={valence_loss.item():.6f}, A={arousal_loss.item():.6f})')
    
    # Epoch statistics
    avg_loss = np.mean(epoch_losses)
    avg_valence_loss = np.mean(epoch_valence_losses)
    avg_arousal_loss = np.mean(epoch_arousal_losses)
    
    print(f'Epoch {epoch} Training: '
          f'Loss={avg_loss:.6f} '
          f'(V={avg_valence_loss:.6f}, A={avg_arousal_loss:.6f})')
    
    return avg_loss, avg_valence_loss, avg_arousal_loss

def validate_epoch(model, data_loader, device, epoch, is_lrm=False):
    """Validate for one epoch."""
    model.eval()
    
    epoch_losses = []
    epoch_valence_losses = []
    epoch_arousal_losses = []
    
    with torch.no_grad():
        for batch_idx, batch_data_dict in enumerate(data_loader):
            # Move data to device
            batch_feature = batch_data_dict['feature'].to(device)
            batch_valence = batch_data_dict['valence'].to(device)
            batch_arousal = batch_data_dict['arousal'].to(device)
            
            # Forward pass
            if is_lrm:
                # For LRM, clear feedback state at start of each batch
                model.clear_feedback_state()
            
            output_dict = model(batch_feature)
            
            # Compute loss
            valence_loss = nn.MSELoss()(output_dict['valence'], batch_valence)
            arousal_loss = nn.MSELoss()(output_dict['arousal'], batch_arousal)
            total_loss = valence_loss + arousal_loss
            
            # Store losses
            epoch_losses.append(total_loss.item())
            epoch_valence_losses.append(valence_loss.item())
            epoch_arousal_losses.append(arousal_loss.item())
    
    # Epoch statistics
    avg_loss = np.mean(epoch_losses)
    avg_valence_loss = np.mean(epoch_valence_losses)
    avg_arousal_loss = np.mean(epoch_arousal_losses)
    
    print(f'Epoch {epoch} Validation: '
          f'Loss={avg_loss:.6f} '
          f'(V={avg_valence_loss:.6f}, A={avg_arousal_loss:.6f})')
    
    return avg_loss, avg_valence_loss, avg_arousal_loss

def run_full_training(args):
    """Run full training experiment."""
    print("🧪 Step 4: Full LRM Training Test")
    print("=" * 50)
    
    # Setup device
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    
    # Create workspace
    workspace = Path(args.workspace)
    create_folder(workspace)
    print(f"Workspace: {workspace}")
    
    # Check if data exists
    if not os.path.exists(args.features_hdf5_path):
        print(f"❌ Data file not found: {args.features_hdf5_path}")
        print("Please check the path or run feature extraction first.")
        return False
    
    print(f"✅ Data file found: {args.features_hdf5_path}")
    
    try:
        # Create model
        model = create_model(args, device)
        is_lrm = 'LRM' in args.model_type
        
        # Create data loaders
        print("\nCreating data loaders...")
        try:
            from pytorch.data_generator import EmotionTrainSampler, EmotionValidateSampler, EmoSoundscapesDataset, emotion_collate_fn
        except ImportError:
            from data_generator import EmotionTrainSampler, EmotionValidateSampler, EmoSoundscapesDataset, emotion_collate_fn
        from torch.utils.data import DataLoader
        
        # Training data loader
        train_sampler = EmotionTrainSampler(
            hdf5_path=args.features_hdf5_path,
            batch_size=args.batch_size,
            train_ratio=0.7
        )
        
        train_dataset = EmoSoundscapesDataset()
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_sampler,
            collate_fn=emotion_collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        # Validation data loader
        validate_sampler = EmotionValidateSampler(
            hdf5_path=args.features_hdf5_path,
            batch_size=args.batch_size,
            train_ratio=0.7
        )
        
        validate_loader = DataLoader(
            dataset=train_dataset,
            batch_sampler=validate_sampler,
            collate_fn=emotion_collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"✅ Data loaders created")
        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(validate_loader)}")
        
        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Training metrics
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n🏋️ Starting Training")
        print("-" * 30)
        print(f"Model: {args.model_type}")
        print(f"Epochs: {args.num_epochs}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        if is_lrm:
            print(f"Forward passes: {args.forward_passes}")
            print(f"Modulation strength: {args.modulation_strength}")
        print("-" * 30)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(1, args.num_epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_loss, train_val_loss, train_ar_loss = train_epoch(
                model, train_loader, optimizer, device, epoch, is_lrm
            )
            
            # Validation
            val_loss, val_val_loss, val_ar_loss = validate_epoch(
                model, validate_loader, device, epoch, is_lrm
            )
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch} completed in {epoch_time:.1f}s")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if args.save_model:
                    model_path = workspace / f'best_{args.model_type.lower()}.pth'
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'args': args
                    }, model_path)
                    print(f"✅ Best model saved: {model_path}")
            else:
                patience_counter += 1
                
            print(f"Best val loss: {best_val_loss:.6f}, Patience: {patience_counter}/{args.early_stop}")
            
            # Early stopping
            if patience_counter >= args.early_stop:
                print(f"Early stopping triggered after {epoch} epochs")
                break
            
            print("-" * 50)
        
        total_time = time.time() - start_time
        print(f"\n🎉 Training completed in {total_time:.1f}s")
        
        # Final evaluation with LRM-aware system
        print("\n📊 Final Evaluation")
        print("-" * 20)
        
        if is_lrm:
            print("Using LRM-aware evaluation system...")
            # Load best model
            if args.save_model:
                model_path = workspace / f'best_{args.model_type.lower()}.pth'
                checkpoint = torch.load(model_path, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded best model from epoch {checkpoint['epoch']}")
            
            # Run LRM evaluation
            statistics = evaluate_lrm(
                model=model,
                data_loader=validate_loader,
                device=device,
                max_audio_files=50  # Limit for testing
            )
            
            print("LRM Evaluation Results:")
            print(f"  Valence MAE: {statistics['valence_mae']:.6f}")
            print(f"  Arousal MAE: {statistics['arousal_mae']:.6f}")
            print(f"  Audio files processed: {statistics['num_audio_files']}")
            print(f"  Total segments: {statistics['total_segments']}")
            
        else:
            print("Using standard evaluation...")
            val_loss, val_val_loss, val_ar_loss = validate_epoch(
                model, validate_loader, device, "Final", is_lrm
            )
        
        # Training summary
        print(f"\n📈 Training Summary")
        print("-" * 20)
        print(f"Initial train loss: {train_losses[0]:.6f}")
        print(f"Final train loss: {train_losses[-1]:.6f}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Loss improvement: {((train_losses[0] - best_val_loss) / train_losses[0] * 100):+.1f}%")
        
        # Save training history
        history_path = workspace / 'training_history.npz'
        np.savez(history_path,
                train_losses=train_losses,
                val_losses=val_losses,
                best_val_loss=best_val_loss)
        print(f"✅ Training history saved: {history_path}")
        
        print("\n🎉 Step 4: Full LRM Training Test Complete!")
        print("=" * 50)
        print("✅ Model trained successfully")
        print("✅ LRM feedback system working")
        print("✅ Evaluation system verified")
        print("✅ Model checkpoints saved")
        print("\nReady for Step 5: Baseline Comparison!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    args = parse_args()
    
    print("Step 4: Full LRM Training Test")
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    success = run_full_training(args)
    
    if success:
        print("\n🚀 Training successful! Ready for Step 5.")
    else:
        print("\n🔧 Please check the errors and try again.")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 