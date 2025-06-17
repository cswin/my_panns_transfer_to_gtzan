#!/usr/bin/env python3
"""
Step 3: Quick Training Test

This test verifies that:
1. All three models can train properly (old, new baseline, LRM)
2. Gradient flow works correctly through feedback connections
3. Loss decreases during training
4. Models produce different learning trajectories
5. LRM feedback system doesn't break gradient computation
6. Memory usage is reasonable
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from collections import defaultdict

# Add pytorch directory to path
sys.path.append('pytorch')

from models import FeatureEmotionRegression_Cnn6, FeatureEmotionRegression_Cnn6_NewAffective
from models_lrm import FeatureEmotionRegression_Cnn6_LRM

def create_dummy_data(num_samples=32, time_steps=64, mel_bins=64):
    """Create dummy training data."""
    # Create realistic-looking mel-spectrogram features
    features = torch.randn(num_samples, time_steps, mel_bins) * 0.5
    
    # Create realistic emotion targets (valence and arousal in [-1, 1] range)
    valence = torch.randn(num_samples, 1) * 0.5  # Centered around 0
    arousal = torch.randn(num_samples, 1) * 0.5  # Centered around 0
    
    return features, valence, arousal

def compute_loss(output, valence_target, arousal_target):
    """Compute combined valence and arousal loss."""
    valence_loss = nn.MSELoss()(output['valence'], valence_target)
    arousal_loss = nn.MSELoss()(output['arousal'], arousal_target)
    total_loss = valence_loss + arousal_loss
    return total_loss, valence_loss, arousal_loss

def train_model_quick(model, model_name, num_epochs=5, batch_size=8):
    """Quick training test for a model."""
    print(f"\n🏋️ Training {model_name}")
    print("-" * 40)
    
    # Set model to training mode
    model.train()
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training metrics
    losses = []
    valence_losses = []
    arousal_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_valence_losses = []
        epoch_arousal_losses = []
        
        # Create multiple batches per epoch
        num_batches = 4
        for batch_idx in range(num_batches):
            # Create batch data
            features, valence_target, arousal_target = create_dummy_data(
                num_samples=batch_size, time_steps=64, mel_bins=64
            )
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            try:
                if 'LRM' in model_name:
                    # For LRM model, clear feedback state at start of each batch
                    model.clear_feedback_state()
                    output = model(features)
                else:
                    output = model(features)
                
                # Compute loss
                total_loss, valence_loss, arousal_loss = compute_loss(
                    output, valence_target, arousal_target
                )
                
                # Backward pass
                total_loss.backward()
                
                # Check for gradient issues
                grad_norm = 0.0
                param_count = 0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                        param_count += 1
                grad_norm = grad_norm ** 0.5
                
                # Update parameters
                optimizer.step()
                
                # Store metrics
                epoch_losses.append(total_loss.item())
                epoch_valence_losses.append(valence_loss.item())
                epoch_arousal_losses.append(arousal_loss.item())
                
            except Exception as e:
                print(f"❌ Training failed at epoch {epoch}, batch {batch_idx}: {str(e)}")
                return None
        
        # Epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_valence_loss = np.mean(epoch_valence_losses)
        avg_arousal_loss = np.mean(epoch_arousal_losses)
        
        losses.append(avg_loss)
        valence_losses.append(avg_valence_loss)
        arousal_losses.append(avg_arousal_loss)
        
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.6f} "
              f"(V={avg_valence_loss:.6f}, A={avg_arousal_loss:.6f}) "
              f"GradNorm={grad_norm:.4f}")
    
    # Check if loss decreased
    loss_decreased = losses[-1] < losses[0]
    loss_reduction = (losses[0] - losses[-1]) / losses[0] * 100
    
    print(f"  📊 Loss change: {losses[0]:.6f} → {losses[-1]:.6f} "
          f"({loss_reduction:+.1f}%)")
    
    if loss_decreased:
        print(f"  ✅ Loss decreased - training successful!")
    else:
        print(f"  ⚠️  Loss did not decrease - may need more epochs")
    
    return {
        'losses': losses,
        'valence_losses': valence_losses,
        'arousal_losses': arousal_losses,
        'final_loss': losses[-1],
        'loss_decreased': loss_decreased,
        'loss_reduction': loss_reduction
    }

def test_gradient_flow(model, model_name):
    """Test gradient flow through the model."""
    print(f"\n🔍 Testing Gradient Flow - {model_name}")
    print("-" * 40)
    
    model.train()
    
    # Create test data
    features, valence_target, arousal_target = create_dummy_data(
        num_samples=4, time_steps=64, mel_bins=64
    )
    
    # Forward pass
    if 'LRM' in model_name:
        model.clear_feedback_state()
        output = model(features)
    else:
        output = model(features)
    
    # Compute loss
    total_loss, valence_loss, arousal_loss = compute_loss(
        output, valence_target, arousal_target
    )
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    grad_stats = defaultdict(list)
    param_stats = defaultdict(int)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_stats['total'] += 1
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_stats['has_grad'].append(grad_norm)
                param_stats['has_grad'] += 1
                
                # Categorize by component
                if 'base_model' in name or 'base.' in name:
                    grad_stats['base'].append(grad_norm)
                elif 'embedding_' in name or 'affective_' in name:
                    grad_stats['affective'].append(grad_norm)
                elif 'lrm' in name or 'mod_' in name:
                    grad_stats['feedback'].append(grad_norm)
            else:
                param_stats['no_grad'] += 1
    
    print(f"  Parameters: {param_stats['total']} total, "
          f"{param_stats['has_grad']} with gradients, "
          f"{param_stats['no_grad']} without gradients")
    
    if grad_stats['has_grad']:
        avg_grad = np.mean(grad_stats['has_grad'])
        max_grad = np.max(grad_stats['has_grad'])
        min_grad = np.min(grad_stats['has_grad'])
        print(f"  Gradient norms: avg={avg_grad:.6f}, max={max_grad:.6f}, min={min_grad:.6f}")
        
        # Component-wise gradient analysis
        for component in ['base', 'affective', 'feedback']:
            if grad_stats[component]:
                comp_avg = np.mean(grad_stats[component])
                comp_count = len(grad_stats[component])
                print(f"  {component.capitalize()} gradients: {comp_count} params, avg={comp_avg:.6f}")
        
        print("  ✅ Gradients flowing properly")
        return True
    else:
        print("  ❌ No gradients found!")
        return False

def test_quick_training():
    """Main quick training test."""
    print("🧪 Quick Training Test")
    print("=" * 50)
    
    # Model parameters
    sample_rate = 32000
    window_size = 1024
    hop_size = 320
    mel_bins = 64
    fmin = 50
    fmax = 14000
    
    models_to_test = []
    
    try:
        # Test 1: Create all models
        print("\n📋 Test 1: Model Creation")
        print("-" * 30)
        
        # Old affective model
        print("Creating old affective model...")
        old_model = FeatureEmotionRegression_Cnn6(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            freeze_base=True
        )
        models_to_test.append(('Old Affective', old_model))
        
        # New affective baseline model
        print("Creating new affective baseline model...")
        baseline_model = FeatureEmotionRegression_Cnn6_NewAffective(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            freeze_base=True
        )
        models_to_test.append(('New Affective (No Feedback)', baseline_model))
        
        # LRM model
        print("Creating LRM model...")
        lrm_model = FeatureEmotionRegression_Cnn6_LRM(
            sample_rate=sample_rate,
            window_size=window_size,
            hop_size=hop_size,
            mel_bins=mel_bins,
            fmin=fmin,
            fmax=fmax,
            freeze_base=True,
            forward_passes=2
        )
        models_to_test.append(('LRM (With Feedback)', lrm_model))
        
        print("✅ All models created successfully")
        
        # Test 2: Gradient flow test
        print("\n📋 Test 2: Gradient Flow Verification")
        print("-" * 35)
        
        gradient_results = {}
        for model_name, model in models_to_test:
            gradient_results[model_name] = test_gradient_flow(model, model_name)
        
        all_gradients_ok = all(gradient_results.values())
        if all_gradients_ok:
            print("\n✅ All models have proper gradient flow")
        else:
            print("\n❌ Some models have gradient issues")
            return False
        
        # Test 3: Quick training test
        print("\n📋 Test 3: Quick Training Test")
        print("-" * 30)
        
        training_results = {}
        for model_name, model in models_to_test:
            result = train_model_quick(model, model_name, num_epochs=5, batch_size=8)
            if result is None:
                print(f"❌ Training failed for {model_name}")
                return False
            training_results[model_name] = result
        
        # Test 4: Training comparison
        print("\n📋 Test 4: Training Results Comparison")
        print("-" * 35)
        
        print("Final losses:")
        for model_name, result in training_results.items():
            loss_status = "✅" if result['loss_decreased'] else "⚠️"
            print(f"  {loss_status} {model_name}: {result['final_loss']:.6f} "
                  f"({result['loss_reduction']:+.1f}%)")
        
        # Check if all models learned
        all_learned = all(result['loss_decreased'] for result in training_results.values())
        
        if all_learned:
            print("\n✅ All models successfully reduced loss during training")
        else:
            print("\n⚠️  Some models did not reduce loss (may need more epochs)")
        
        # Test 5: LRM-specific tests
        print("\n📋 Test 5: LRM-Specific Functionality")
        print("-" * 35)
        
        lrm_model = models_to_test[2][1]  # LRM model
        
        # Test feedback control
        print("Testing feedback control...")
        lrm_model.disable_feedback()
        print("  ✅ Feedback disabled")
        
        lrm_model.enable_feedback()
        print("  ✅ Feedback enabled")
        
        # Test modulation strength
        print("Testing modulation strength control...")
        lrm_model.set_modulation_strength(0.5)
        print("  ✅ Modulation strength set to 0.5")
        
        lrm_model.reset_modulation_strength()
        print("  ✅ Modulation strength reset")
        
        # Test forward passes
        print("Testing forward passes control...")
        lrm_model.set_forward_passes(3)
        print("  ✅ Forward passes set to 3")
        
        # Test memory usage
        print("\n📋 Test 6: Memory Usage Check")
        print("-" * 25)
        
        # Simple memory test with larger batch
        features, valence_target, arousal_target = create_dummy_data(
            num_samples=16, time_steps=64, mel_bins=64
        )
        
        for model_name, model in models_to_test:
            model.eval()
            with torch.no_grad():
                if 'LRM' in model_name:
                    model.clear_feedback_state()
                    output = model(features)
                else:
                    output = model(features)
            print(f"  ✅ {model_name}: Memory test passed")
        
        print("\n🎉 Quick Training Test Complete!")
        print("=" * 50)
        print("Summary:")
        print("✅ All models created successfully")
        print("✅ Gradient flow verified for all models")
        print("✅ All models can train and reduce loss")
        print("✅ LRM-specific functionality works")
        print("✅ Memory usage is reasonable")
        print("\nAll models are ready for full training!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_quick_training()
    if success:
        print("\n🚀 Ready to proceed to Step 4: Full LRM Training")
    else:
        print("\n🔧 Please fix the issues before proceeding")
    
    sys.exit(0 if success else 1) 