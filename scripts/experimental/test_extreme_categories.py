#!/usr/bin/env python3
"""
Test with extremely different emotion categories to see if LRM can differentiate them.
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

def test_extreme_categories():
    """Test with extremely different emotion categories."""
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
    
    # Test EXTREMELY different categories
    test_categories = [
        # From 9-bin system
        ('9-bin', 'negative_strong', signals_9bin),
        ('9-bin', 'positive_strong', signals_9bin),
        ('9-bin', 'neutral_middle', signals_9bin),
        # From 25-bin system  
        ('25-bin', 'very_negative_very_strong', signals_25bin),
        ('25-bin', 'very_positive_very_strong', signals_25bin),
        ('25-bin', 'neutral_middle', signals_25bin),
        ('25-bin', 'very_negative_very_weak', signals_25bin),
        ('25-bin', 'very_positive_very_weak', signals_25bin),
    ]
    
    print(f"\n📊 TESTING EXTREME CATEGORIES (Strength=5.0)")
    print(f"{'System':<8} {'Category':<25} {'Valence':<12} {'Arousal':<12} {'ΔV':<12} {'ΔA':<12}")
    print("-" * 85)
    
    results = {}
    strength = 5.0
    
    for system, category, signals_dict in test_categories:
        if category not in signals_dict:
            print(f"⚠️  {system} {category}: NOT FOUND")
            continue
            
        signals = signals_dict[category]
        
        # Apply steering signals
        model.clear_feedback_state()
        model.lrm.mod_inputs.clear()
        
        # Pre-populate mod_inputs with steering signals
        if 'valence_128d' in signals:
            valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
            model.lrm.mod_inputs['from_affective_valence_128d_to_visual_system_base_conv_block4'] = valence_signal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            model.lrm.mod_inputs['from_affective_valence_128d_to_visual_system_base_conv_block3'] = valence_signal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        if 'arousal_128d' in signals:
            arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
            model.lrm.mod_inputs['from_affective_arousal_128d_to_visual_system_base_conv_block2'] = arousal_signal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            model.lrm.mod_inputs['from_affective_arousal_128d_to_visual_system_base_conv_block1'] = arousal_signal.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # Apply modulation
        model.lrm.adjust_modulation_strength(strength)
        model.lrm.enable()
        
        with torch.no_grad():
            output = model(sample_tensor, forward_passes=2)
        
        v_out = output['valence'].cpu().item()
        a_out = output['arousal'].cpu().item()
        delta_v = v_out - baseline_v
        delta_a = a_out - baseline_a
        
        results[f"{system}_{category}"] = {'valence': v_out, 'arousal': a_out}
        
        print(f"{system:<8} {category:<25} {v_out:<12.6f} {a_out:<12.6f} {delta_v:<12.6f} {delta_a:<12.6f}")
        
        model.lrm.reset_modulation_strength()
    
    # Analysis of differences
    print(f"\n🔍 DIFFERENCE ANALYSIS:")
    categories = list(results.keys())
    for i in range(len(categories)):
        for j in range(i+1, len(categories)):
            cat1, cat2 = categories[i], categories[j]
            v1, a1 = results[cat1]['valence'], results[cat1]['arousal']
            v2, a2 = results[cat2]['valence'], results[cat2]['arousal']
            
            diff_v = abs(v1 - v2)
            diff_a = abs(a1 - a2)
            
            if diff_v > 0.001 or diff_a > 0.001:  # Only show meaningful differences
                print(f"✅ {cat1} vs {cat2}:")
                print(f"   ΔV = {diff_v:.6f}, ΔA = {diff_a:.6f}")
    
    # Check signal value differences
    print(f"\n🔍 SIGNAL VALUE ANALYSIS:")
    if 'negative_strong' in signals_9bin and 'positive_strong' in signals_9bin:
        neg_val = np.array(signals_9bin['negative_strong']['valence_128d'])
        pos_val = np.array(signals_9bin['positive_strong']['valence_128d'])
        diff = np.abs(neg_val - pos_val)
        print(f"9-bin negative_strong vs positive_strong valence diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    if 'very_negative_very_strong' in signals_25bin and 'very_positive_very_strong' in signals_25bin:
        neg_val = np.array(signals_25bin['very_negative_very_strong']['valence_128d'])
        pos_val = np.array(signals_25bin['very_positive_very_strong']['valence_128d'])
        diff = np.abs(neg_val - pos_val)
        print(f"25-bin very_negative_very_strong vs very_positive_very_strong valence diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    print(f"\n🎯 FINAL CONCLUSION:")
    print(f"If ALL different categories produce identical outputs, the LRM system has a fundamental issue.")
    print(f"If SOME categories produce different outputs, then the system works but needs proper category selection.")

if __name__ == "__main__":
    test_extreme_categories() 