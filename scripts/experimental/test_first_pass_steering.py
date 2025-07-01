#!/usr/bin/env python3
"""
Test first_pass_steering to fix the signal convergence issue.
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

def test_first_pass_steering():
    """Test if first_pass_steering fixes the convergence issue."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load model
    checkpoint_path = 'workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
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
    
    # Test the CORRECTED approach: Apply steering signals BEFORE forward pass
    print(f"\n📊 TESTING CORRECTED STEERING (Pre-applied signals)")
    print(f"{'Method':<15} {'Strength':<8} {'Valence':<12} {'Arousal':<12} {'ΔV':<12} {'ΔA':<12}")
    print("-" * 75)
    
    strengths = [1.0, 2.0, 5.0]
    
    for strength in strengths:
        results = {}
        
        # Test 9-bin method
        category_9bin = 'negative_strong'
        if category_9bin in signals_9bin:
            signals = signals_9bin[category_9bin]
            
            # CRITICAL FIX: Clear and apply signals BEFORE forward pass
            model.clear_feedback_state()
            model.lrm.mod_inputs.clear()  # Ensure clean state
            
            # Pre-populate mod_inputs with steering signals
            if 'valence_128d' in signals:
                valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
                # Add to mod_inputs directly for immediate availability
                model.lrm.mod_inputs['from_affective_valence_128d_to_visual_system_base_conv_block4'] = valence_signal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                model.lrm.mod_inputs['from_affective_valence_128d_to_visual_system_base_conv_block3'] = valence_signal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
            if 'arousal_128d' in signals:
                arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
                # Add to mod_inputs directly for immediate availability  
                model.lrm.mod_inputs['from_affective_arousal_128d_to_visual_system_base_conv_block2'] = arousal_signal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                model.lrm.mod_inputs['from_affective_arousal_128d_to_visual_system_base_conv_block1'] = arousal_signal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
            # Apply modulation strength
            model.lrm.adjust_modulation_strength(strength)
            model.lrm.enable()
            
            with torch.no_grad():
                output = model(sample_tensor, forward_passes=2)
            
            results['9bin'] = {
                'valence': output['valence'].cpu().item(),
                'arousal': output['arousal'].cpu().item()
            }
            
            model.lrm.reset_modulation_strength()
        
        # Test 25-bin method
        category_25bin = 'very_negative_very_strong'
        if category_25bin in signals_25bin:
            signals = signals_25bin[category_25bin]
            
            # CRITICAL FIX: Clear and apply signals BEFORE forward pass
            model.clear_feedback_state()
            model.lrm.mod_inputs.clear()  # Ensure clean state
            
            # Pre-populate mod_inputs with steering signals
            if 'valence_128d' in signals:
                valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
                # Add to mod_inputs directly for immediate availability
                model.lrm.mod_inputs['from_affective_valence_128d_to_visual_system_base_conv_block4'] = valence_signal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                model.lrm.mod_inputs['from_affective_valence_128d_to_visual_system_base_conv_block3'] = valence_signal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
            if 'arousal_128d' in signals:
                arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
                # Add to mod_inputs directly for immediate availability
                model.lrm.mod_inputs['from_affective_arousal_128d_to_visual_system_base_conv_block2'] = arousal_signal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                model.lrm.mod_inputs['from_affective_arousal_128d_to_visual_system_base_conv_block1'] = arousal_signal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            
            # Apply modulation strength
            model.lrm.adjust_modulation_strength(strength)
            model.lrm.enable()
            
            with torch.no_grad():
                output = model(sample_tensor, forward_passes=2)
            
            results['25bin'] = {
                'valence': output['valence'].cpu().item(),
                'arousal': output['arousal'].cpu().item()
            }
            
            model.lrm.reset_modulation_strength()
        
        # Calculate differences
        if '9bin' in results and '25bin' in results:
            v_9bin = results['9bin']['valence']
            a_9bin = results['9bin']['arousal']
            v_25bin = results['25bin']['valence']
            a_25bin = results['25bin']['arousal']
            
            delta_v_9bin = v_9bin - baseline_v
            delta_a_9bin = a_9bin - baseline_a
            delta_v_25bin = v_25bin - baseline_v
            delta_a_25bin = a_25bin - baseline_a
            
            print(f"{'9-bin':<15} {strength:<8.1f} {v_9bin:<12.6f} {a_9bin:<12.6f} {delta_v_9bin:<12.6f} {delta_a_9bin:<12.6f}")
            print(f"{'25-bin':<15} {strength:<8.1f} {v_25bin:<12.6f} {a_25bin:<12.6f} {delta_v_25bin:<12.6f} {delta_a_25bin:<12.6f}")
            
            diff_v = abs(v_9bin - v_25bin)
            diff_a = abs(a_9bin - a_25bin)
            print(f"{'Difference':<15} {'':<8} {diff_v:<12.6f} {diff_a:<12.6f}")
            print("-" * 75)
    
    print(f"\n🔍 ANALYSIS:")
    print(f"✅ If differences are now non-zero, the fix works!")
    print(f"✅ Different steering signals should produce different outputs")
    print(f"✅ This proves the 25-bin system can provide more granular control")

if __name__ == "__main__":
    test_first_pass_steering() 