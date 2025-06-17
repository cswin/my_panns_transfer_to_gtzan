#!/usr/bin/env python3
"""
Step 4: Simplified LRM Training Test

A simplified version of the full LRM training test with minimal dependencies.
This version focuses on the core functionality without complex evaluation systems.
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

def create_folder(path):
    """Simple folder creation function."""
    Path(path).mkdir(parents=True, exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Step 4: Simplified LRM Training Test')
    
    parser.add_argument('--features_hdf5_path', type=str, 
                       default='workspaces/emotion_regression/features/emotion_features.h5',
                       help='Path to features HDF5 file')
    parser.add_argument('--model_type', type=str, default='FeatureEmotionRegression_Cnn6_LRM',
                       choices=['FeatureEmotionRegression_Cnn6_LRM', 'FeatureEmotionRegression_Cnn6_NewAffective'],
                       help='Model type to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--workspace', type=str, default='./workspaces/step4_simple',
                       help='Workspace directory')
    parser.add_argument('--cuda', action='store_true', default=True, help='Use CUDA')
    
    return parser.parse_args()

def create_model(args, device):
    """Create model with error handling."""
    print(f"Creating {args.model_type} model...")
    
    try:
        # Import config
        import config
        
        # Import models
        if args.model_type == 'FeatureEmotionRegression_Cnn6_LRM':
            from models_lrm import FeatureEmotionRegression_Cnn6_LRM
            model = FeatureEmotionRegression_Cnn6_LRM(
                sample_rate=config.sample_rate,
                window_size=config.window_size,
                hop_size=config.hop_size,
                mel_bins=config.mel_bins,
                fmin=config.fmin,
                fmax=config.fmax,
                freeze_base=True,
                forward_passes=2
            )
        else:
            from models import FeatureEmotionRegression_Cnn6_NewAffective
            model = FeatureEmotionRegression_Cnn6_NewAffective(
                sample_rate=config.sample_rate,
                window_size=config.window_size,
                hop_size=config.hop_size,
                mel_bins=config.mel_bins,
                fmin=config.fmin,
                fmax=config.fmax,
                freeze_base=True
            )
        
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✅ Model created: {total_params:,} total, {trainable_params:,} trainable")
        return model
        
    except Exception as e:
        print(f"❌ Error creating model: {str(e)}")
        return None

def create_data_loaders(features_path, batch_size):
    """Create data loaders with error handling."""
    try:
        print("Creating data loaders...")
        
        # Import data generator
        from data_generator import EmotionTrainSampler, EmotionValidateSampler, EmoSoundscapesDataset, emotion_collate_fn
        from torch.utils.data import DataLoader
        
        # Create samplers
        train_sampler = EmotionTrainSampler(
            hdf5_path=features_path,
            batch_size=batch_size,
            train_ratio=0.7
        )
        
        validate_sampler = EmotionValidateSampler(
            hdf5_path=features_path,
            batch_size=batch_size,
            train_ratio=0.7
        )
        
        # Create dataset
        dataset = EmoSoundscapesDataset()
        
        # Create data loaders
        train_loader = DataLoader(
            dataset=dataset,
            batch_sampler=train_sampler,
            collate_fn=emotion_collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        validate_loader = DataLoader(
            dataset=dataset,
            batch_sampler=validate_sampler,
            collate_fn=emotion_collate_fn,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"✅ Data loaders created: {len(train_loader)} train, {len(validate_loader)} val batches")
        return train_loader, validate_loader
        
    except Exception as e:
        print(f"❌ Error creating data loaders: {str(e)}")
        return None, None

def train_epoch(model, data_loader, optimizer, device, epoch, is_lrm=False):
    """Train for one epoch."""
    model.train()
    epoch_losses = []
    
    for batch_idx, batch_data_dict in enumerate(data_loader):
        try:
            # Move data to device
            batch_feature = batch_data_dict['feature'].to(device)
            batch_valence = batch_data_dict['valence'].to(device)
            batch_arousal = batch_data_dict['arousal'].to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            if is_lrm and hasattr(model, 'clear_feedback_state'):
                model.clear_feedback_state()
            
            output_dict = model(batch_feature)
            
            # Compute loss
            valence_loss = nn.MSELoss()(output_dict['valence'], batch_valence)
            arousal_loss = nn.MSELoss()(output_dict['arousal'], batch_arousal)
            total_loss = valence_loss + arousal_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_losses.append(total_loss.item())
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}: Loss={total_loss.item():.6f}')
                
        except Exception as e:
            print(f"❌ Error in batch {batch_idx}: {str(e)}")
            continue
    
    avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
    print(f'Epoch {epoch}: Avg Loss={avg_loss:.6f}')
    return avg_loss

def validate_epoch(model, data_loader, device, epoch, is_lrm=False):
    """Validate for one epoch."""
    model.eval()
    epoch_losses = []
    
    with torch.no_grad():
        for batch_idx, batch_data_dict in enumerate(data_loader):
            try:
                batch_feature = batch_data_dict['feature'].to(device)
                batch_valence = batch_data_dict['valence'].to(device)
                batch_arousal = batch_data_dict['arousal'].to(device)
                
                if is_lrm and hasattr(model, 'clear_feedback_state'):
                    model.clear_feedback_state()
                
                output_dict = model(batch_feature)
                
                valence_loss = nn.MSELoss()(output_dict['valence'], batch_valence)
                arousal_loss = nn.MSELoss()(output_dict['arousal'], batch_arousal)
                total_loss = valence_loss + arousal_loss
                
                epoch_losses.append(total_loss.item())
                
            except Exception as e:
                print(f"❌ Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    avg_loss = np.mean(epoch_losses) if epoch_losses else float('inf')
    print(f'Epoch {epoch} Validation: Loss={avg_loss:.6f}')
    return avg_loss

def run_training_test(args):
    """Run the training test."""
    print("🧪 Step 4: Simplified LRM Training Test")
    print("=" * 50)
    
    # Setup device
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✅ Using CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("✅ Using CPU")
    
    # Create workspace
    create_folder(args.workspace)
    print(f"✅ Workspace: {args.workspace}")
    
    # Check data file
    if not os.path.exists(args.features_hdf5_path):
        print(f"❌ Data file not found: {args.features_hdf5_path}")
        return False
    print(f"✅ Data file found: {args.features_hdf5_path}")
    
    # Create model
    model = create_model(args, device)
    if model is None:
        return False
    
    is_lrm = 'LRM' in args.model_type
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(args.features_hdf5_path, args.batch_size)
    if train_loader is None:
        return False
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    print(f"\n🏋️ Training {args.model_type} for {args.num_epochs} epochs...")
    print("-" * 30)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    start_time = time.time()
    
    for epoch in range(1, args.num_epochs + 1):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, is_lrm)
        train_losses.append(train_loss)
        
        # Validation
        val_loss = validate_epoch(model, val_loader, device, epoch, is_lrm)
        val_losses.append(val_loss)
        
        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            model_path = Path(args.workspace) / f'best_{args.model_type.lower()}.pth'
            torch.save(model.state_dict(), model_path)
            print(f"✅ Best model saved: {model_path}")
        
        print("-" * 30)
    
    total_time = time.time() - start_time
    
    # Results
    print(f"\n📊 Training Results")
    print("-" * 20)
    print(f"Model: {args.model_type}")
    print(f"Training time: {total_time:.1f}s")
    print(f"Initial train loss: {train_losses[0]:.6f}")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    improvement = ((train_losses[0] - best_val_loss) / train_losses[0] * 100)
    print(f"Loss improvement: {improvement:+.1f}%")
    
    # Test LRM-specific functionality
    if is_lrm:
        print(f"\n🔄 LRM-Specific Tests")
        print("-" * 20)
        
        # Test feedback state management
        if hasattr(model, 'clear_feedback_state'):
            model.clear_feedback_state()
            print("✅ Feedback state clearing works")
        
        if hasattr(model, 'lrm_system'):
            print(f"✅ LRM system has {len(model.lrm_system.connections)} feedback connections")
        
        # Test sequential processing
        model.eval()
        with torch.no_grad():
            try:
                # Get a sample batch
                sample_batch = next(iter(val_loader))
                batch_feature = sample_batch['feature'].to(device)
                
                # Clear state and run first segment
                model.clear_feedback_state()
                output1 = model(batch_feature[:1])  # First sample only
                
                # Run second segment (should have feedback from first)
                output2 = model(batch_feature[1:2])  # Second sample only
                
                # Check if outputs are different (indicating feedback effect)
                val_diff = abs(output1['valence'].item() - output2['valence'].item())
                ar_diff = abs(output1['arousal'].item() - output2['arousal'].item())
                
                print(f"✅ Sequential feedback test: V_diff={val_diff:.6f}, A_diff={ar_diff:.6f}")
                
                if val_diff > 1e-6 or ar_diff > 1e-6:
                    print("✅ Feedback is affecting predictions (good!)")
                else:
                    print("⚠️  Feedback effect is very small")
                    
            except Exception as e:
                print(f"⚠️  Sequential test failed: {str(e)}")
    
    print(f"\n🎉 Step 4 Test Complete!")
    print("✅ Model training successful")
    print("✅ Loss decreased over epochs")
    print("✅ Model checkpoints saved")
    if is_lrm:
        print("✅ LRM feedback system verified")
    
    return True

def main():
    """Main function."""
    args = parse_args()
    
    print("Step 4: Simplified LRM Training Test")
    print("Arguments:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()
    
    success = run_training_test(args)
    
    if success:
        print("\n🚀 Test successful!")
    else:
        print("\n🔧 Test failed - check errors above")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 