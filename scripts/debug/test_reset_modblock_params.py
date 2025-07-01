#!/usr/bin/env python3
"""
Test script to reset ModBlock parameters after loading trained checkpoint
"""

import os
import sys
import torch
import json
import h5py
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from src.models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
from src.utils.config import cnn6_config, sample_rate

def test_reset_modblock_params():
    """Test if resetting ModBlock parameters improves steering."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=sample_rate,
        window_size=cnn6_config['window_size'],
        hop_size=cnn6_config['hop_size'],
        mel_bins=cnn6_config['mel_bins'],
        fmin=cnn6_config['fmin'],
        fmax=cnn6_config['fmax'],
        freeze_base=True,
        forward_passes=2
    ).to(device)
    
    # Load trained checkpoint
    checkpoint_paths = [
        '/home/pengliu/Private/my_panns_transfer_to_gtzan/workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth',
        'workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if not checkpoint_path:
        print("❌ Error: No checkpoint found")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("✅ Loaded trained model")
    
    # Load steering signals
    steering_signals_path = './steering_signals_25bin/steering_signals_25bin.json'
    if not os.path.exists(steering_signals_path):
        steering_signals_path = './tmp/25bin_steering_signals/steering_signals_25bin.json'
    
    with open(steering_signals_path, 'r') as f:
        signals_25bin = json.load(f)
    
    # Load test sample
    dataset_paths = [
        'workspaces/emotion_regression/features/emotion_features.h5',
        '/DATA/pliu/EmotionData/emotion_features.h5',
        './features/emotion_features.h5'
    ]
    
    test_feature = None
    for path in dataset_paths:
        if os.path.exists(path):
            with h5py.File(path, 'r') as hf:
                test_feature = torch.tensor(hf['feature'][0], dtype=torch.float32).unsqueeze(0).to(device)
            break
    
    if test_feature is None:
        print("❌ Error: Could not load test data")
        return
    
    print(f"✅ Loaded test feature: {test_feature.shape}")
    
    # Test 1: Baseline prediction
    print("\n🧪 Test 1: Baseline (no steering)")
    with torch.no_grad():
        baseline_output = model(test_feature, forward_passes=2, steering_signals=None, first_pass_steering=False)
        baseline_val = baseline_output['valence'][0].item()
        baseline_aro = baseline_output['arousal'][0].item()
        print(f"   Baseline: Valence={baseline_val:.4f}, Arousal={baseline_aro:.4f}")
    
    # Test 2: Steering with trained parameters
    category = 'very_negative_strong'
    signals = signals_25bin[category]
    
    print(f"\n🧪 Test 2: Steering with trained ModBlock parameters")
    
    # Prepare steering signals
    steering_signals_current = []
    
    if 'valence_128d' in signals:
        valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
        steering_signals_current.append({
            'source': 'affective_valence_128d',
            'activation': valence_signal,
            'strength': 10.0,
            'alpha': 1.0
        })
    
    if 'arousal_128d' in signals:
        arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
        steering_signals_current.append({
            'source': 'affective_arousal_128d',
            'activation': arousal_signal,
            'strength': 10.0,
            'alpha': 1.0
        })
    
    with torch.no_grad():
        steered_output = model(test_feature, 
                             forward_passes=2,
                             steering_signals=steering_signals_current,
                             first_pass_steering=False)
        
        steered_val = steered_output['valence'][0].item()
        steered_aro = steered_output['arousal'][0].item()
        
        val_change = steered_val - baseline_val
        aro_change = steered_aro - baseline_aro
        
        print(f"   Trained params: Val={steered_val:.4f} (Δ={val_change:+.4f}), Aro={steered_aro:.4f} (Δ={aro_change:+.4f})")
    
    # Test 3: Reset ModBlock parameters and test again
    print(f"\n🧪 Test 3: Reset ModBlock parameters to original values")
    
    # Reset ModBlock parameters to their initialization values
    for name, lrm_module in model.lrm.named_children():
        if hasattr(lrm_module, 'neg_scale_orig') and hasattr(lrm_module, 'pos_scale_orig'):
            # Reset to original values
            lrm_module.neg_scale.data = lrm_module.neg_scale_orig.clone()
            lrm_module.pos_scale.data = lrm_module.pos_scale_orig.clone()
            print(f"   Reset {name}: neg_scale={lrm_module.neg_scale.item():.4f}, pos_scale={lrm_module.pos_scale.item():.4f}")
        else:
            print(f"   ⚠️  {name}: No original parameters found")
    
    # Test steering with reset parameters
    print(f"\n🧪 Test 4: Steering with reset ModBlock parameters")
    
    with torch.no_grad():
        steered_output_reset = model(test_feature, 
                                   forward_passes=2,
                                   steering_signals=steering_signals_current,
                                   first_pass_steering=False)
        
        steered_val_reset = steered_output_reset['valence'][0].item()
        steered_aro_reset = steered_output_reset['arousal'][0].item()
        
        val_change_reset = steered_val_reset - baseline_val
        aro_change_reset = steered_aro_reset - baseline_val
        
        print(f"   Reset params: Val={steered_val_reset:.4f} (Δ={val_change_reset:+.4f}), Aro={steered_aro_reset:.4f} (Δ={aro_change_reset:+.4f})")
    
    # Test 5: Try different strengths with reset parameters
    print(f"\n🧪 Test 5: Different strengths with reset parameters")
    
    strengths = [1.0, 5.0, 10.0, 20.0, 50.0]
    
    for strength in strengths:
        # Update strength in steering signals
        for signal in steering_signals_current:
            signal['strength'] = strength
        
        with torch.no_grad():
            output = model(test_feature, 
                         forward_passes=2,
                         steering_signals=steering_signals_current,
                         first_pass_steering=False)
            
            val = output['valence'][0].item()
            aro = output['arousal'][0].item()
            
            val_change = val - baseline_val
            aro_change = aro - baseline_aro
            
            print(f"   Strength {strength:5.1f}: Val={val:.4f} (Δ={val_change:+.4f}), Aro={aro:.4f} (Δ={aro_change:+.4f})")
    
    print(f"\n📊 Summary:")
    print(f"   Trained params effect: Val Δ={val_change:+.4f}, Aro Δ={aro_change:+.4f}")
    print(f"   Reset params effect: Val Δ={val_change_reset:+.4f}, Aro Δ={aro_change_reset:+.4f}")
    
    if abs(val_change_reset) > abs(val_change) or abs(aro_change_reset) > abs(aro_change):
        print("   ✅ Reset parameters show stronger steering effects")
    else:
        print("   ❌ Reset parameters don't improve steering")

if __name__ == "__main__":
    test_reset_modblock_params() 