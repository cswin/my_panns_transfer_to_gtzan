#!/usr/bin/env python3
"""
Test different steering signal strengths to find optimal differentiation between methods.
"""

import os
import sys
import torch
import numpy as np
import json
import h5py

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def load_emotion_model(checkpoint_path, device):
    """Load the emotion model with LRM capabilities."""
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        freeze_base=True
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model

def test_strength_sensitivity():
    """Test different steering strengths to find optimal differentiation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load model
    checkpoint_path = 'workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    model = load_emotion_model(checkpoint_path, device)
    
    # Load dataset
    dataset_path = 'workspaces/emotion_feedback/features/emotion_features.h5'
    with h5py.File(dataset_path, 'r') as hf:
        features = hf['feature'][:]
        valence_targets = hf['valence'][:]
        arousal_targets = hf['arousal'][:]
    
    # Use validation split
    np.random.seed(42)
    total_samples = len(features)
    indices = np.random.permutation(total_samples)
    train_size = int(total_samples * 0.7)
    val_indices = indices[train_size:]
    
    features = features[val_indices]
    valence_targets = valence_targets[val_indices]
    arousal_targets = arousal_targets[val_indices]
    
    # Load steering signals
    with open('tmp/steering_signals_by_category.json', 'r') as f:
        signals_9bin = json.load(f)
    with open('tmp/25bin_steering_signals/steering_signals_25bin.json', 'r') as f:
        signals_25bin = json.load(f)
    
    # Test sample
    sample_idx = 0
    sample_tensor = torch.tensor(features[sample_idx:sample_idx+1], dtype=torch.float32).to(device)
    target_v = valence_targets[sample_idx]
    target_a = arousal_targets[sample_idx]
    
    print(f"📊 Target: V={target_v:.3f}, A={target_a:.3f}")
    
    # Get baseline
    model.clear_feedback_state()
    with torch.no_grad():
        baseline_output = model(sample_tensor)
    baseline_v = baseline_output['valence'].cpu().item()
    baseline_a = baseline_output['arousal'].cpu().item()
    print(f"🎯 Baseline: V={baseline_v:.6f}, A={baseline_a:.6f}")
    
    # Test different strengths
    strengths = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    
    print(f"\n📊 STRENGTH SENSITIVITY TEST")
    print(f"{'Strength':<8} {'9bin-V':<12} {'9bin-A':<12} {'25bin-V':<12} {'25bin-A':<12} {'Diff-V':<12} {'Diff-A':<12}")
    print("-" * 85)
    
    for strength in strengths:
        results = {}
        
        # Test 9-bin
        category_9bin = 'negative_strong'
        if category_9bin in signals_9bin:
            signals = signals_9bin[category_9bin]
            
            model.clear_feedback_state()
            
            if 'valence_128d' in signals:
                model.add_steering_signal(
                    source='affective_valence_128d',
                    activation=torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device),
                    strength=strength,
                    alpha=1.0
                )
            
            if 'arousal_128d' in signals:
                model.add_steering_signal(
                    source='affective_arousal_128d',
                    activation=torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device),
                    strength=strength,
                    alpha=1.0
                )
            
            model.lrm.enable()
            
            with torch.no_grad():
                output = model(sample_tensor, forward_passes=2)
            results['9bin'] = {
                'valence': output['valence'].cpu().item(),
                'arousal': output['arousal'].cpu().item()
            }
        
        # Test 25-bin
        category_25bin = 'very_negative_very_strong'
        if category_25bin in signals_25bin:
            signals = signals_25bin[category_25bin]
            
            model.clear_feedback_state()
            
            if 'valence_128d' in signals:
                model.add_steering_signal(
                    source='affective_valence_128d',
                    activation=torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device),
                    strength=strength,
                    alpha=1.0
                )
            
            if 'arousal_128d' in signals:
                model.add_steering_signal(
                    source='affective_arousal_128d',
                    activation=torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device),
                    strength=strength,
                    alpha=1.0
                )
            
            model.lrm.enable()
            
            with torch.no_grad():
                output = model(sample_tensor, forward_passes=2)
            results['25bin'] = {
                'valence': output['valence'].cpu().item(),
                'arousal': output['arousal'].cpu().item()
            }
        
        # Calculate differences
        if '9bin' in results and '25bin' in results:
            v_9bin = results['9bin']['valence']
            a_9bin = results['9bin']['arousal']
            v_25bin = results['25bin']['valence']
            a_25bin = results['25bin']['arousal']
            
            diff_v = abs(v_9bin - v_25bin)
            diff_a = abs(a_9bin - a_25bin)
            
            print(f"{strength:<8.1f} {v_9bin:<12.6f} {a_9bin:<12.6f} {v_25bin:<12.6f} {a_25bin:<12.6f} {diff_v:<12.6f} {diff_a:<12.6f}")
    
    print(f"\n🔍 ANALYSIS:")
    print(f"- Look for strength values where Diff-V and Diff-A are maximized")
    print(f"- Very small differences suggest saturation or insufficient sensitivity")
    print(f"- Optimal strength should show clear differentiation between methods")

if __name__ == "__main__":
    test_strength_sensitivity() 