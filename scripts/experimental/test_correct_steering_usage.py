#!/usr/bin/env python3
"""
Test using the CORRECT steering signals approach from the working sample code.
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

def test_correct_steering_usage():
    """Test using the correct steering signals parameter as shown in working sample."""
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
    with torch.no_grad():
        baseline_output = model(sample_tensor)
    baseline_v = baseline_output['valence'].cpu().item()
    baseline_a = baseline_output['arousal'].cpu().item()
    print(f"🎯 Baseline: V={baseline_v:.6f}, A={baseline_a:.6f}")
    
    # Test with CORRECT steering approach (like working sample)
    print(f"\n📊 TESTING CORRECT STEERING APPROACH")
    print(f"{'Method':<15} {'Valence':<12} {'Arousal':<12} {'ΔV':<12} {'ΔA':<12}")
    print("-" * 65)
    
    test_cases = [
        ('9-bin neg_strong', 'negative_strong', signals_9bin),
        ('9-bin pos_strong', 'positive_strong', signals_9bin),
        ('25-bin very_neg', 'very_negative_very_strong', signals_25bin),
        ('25-bin neutral', 'neutral_middle', signals_25bin),
    ]
    
    for method_name, category, signals_dict in test_cases:
        if category not in signals_dict:
            print(f"⚠️ {method_name}: {category} not found")
            continue
        
        signals = signals_dict[category]
        
        # CORRECT APPROACH: Use steering_signals parameter like working sample
        steering_signals_current = []
        
        if 'valence_128d' in signals:
            valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
            steering_signals_current.append({
                'source': 'affective_valence_128d',
                'activation': valence_signal,
                'strength': 5.0,
                'alpha': 1.0
            })
        
        if 'arousal_128d' in signals:
            arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
            steering_signals_current.append({
                'source': 'affective_arousal_128d', 
                'activation': arousal_signal,
                'strength': 5.0,
                'alpha': 1.0
            })
        
        # Use the CORRECT model forward call (like working sample)
        with torch.no_grad():
            output = model(
                sample_tensor,
                forward_passes=2,
                steering_signals=steering_signals_current,
                first_pass_steering=True,  # KEY: Enable first pass steering
                return_list=False
            )
        
        v_out = output['valence'].cpu().item()
        a_out = output['arousal'].cpu().item()
        delta_v = v_out - baseline_v
        delta_a = a_out - baseline_a
        
        print(f"{method_name:<15} {v_out:<12.6f} {a_out:<12.6f} {delta_v:<12.6f} {delta_a:<12.6f}")
    
    print(f"\n🎯 ANALYSIS:")
    print(f"If different methods now produce different outputs, the correct approach works!")
    print(f"Key insight: Use steering_signals parameter with first_pass_steering=True")

if __name__ == "__main__":
    test_correct_steering_usage() 