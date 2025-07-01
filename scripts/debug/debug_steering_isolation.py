#!/usr/bin/env python3

import sys
import os
# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import json
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def test_steering_isolation():
    """Minimal test to isolate steering issue."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, forward_passes=2
    ).to(device)
    
    # Load pretrained weights
    checkpoint_path = '/DATA/pliu/EmotionData/Cnn6_mAP=0.343.pth'
    if os.path.exists(checkpoint_path):
        model.load_from_pretrain(checkpoint_path)
        print("✅ Loaded pretrained weights")
    
    # Load steering signals
    steering_signals_path = './steering_signals_25bin/steering_signals_25bin.json'
    if not os.path.exists(steering_signals_path):
        steering_signals_path = './tmp/25bin_steering_signals/steering_signals_25bin.json'
    
    with open(steering_signals_path, 'r') as f:
        signals_25bin = json.load(f)
    
    # Get first available category
    category = list(signals_25bin.keys())[0]
    signals = signals_25bin[category]
    print(f"🎯 Using category: {category}")
    
    # Test sample
    sample_tensor = torch.randn(1, 1024, 64).to(device)
    
    print("\n" + "="*60)
    print("STEERING ISOLATION TEST")
    print("="*60)
    
    # Test 1: Baseline (no steering)
    print("\n🔍 Test 1: Baseline (no steering)")
    model.lrm.clear_stored_activations()
    model.lrm.disable()
    
    with torch.no_grad():
        baseline_output = model(sample_tensor, forward_passes=2)
    
    baseline_val = baseline_output['valence'].item()
    baseline_aro = baseline_output['arousal'].item()
    print(f"   Baseline: Valence={baseline_val:.6f}, Arousal={baseline_aro:.6f}")
    
    # Test 2: With steering (exactly like real emotion test)
    print("\n🔍 Test 2: With steering (strength=5.0, like real emotion test)")
    model.lrm.clear_stored_activations()
    
    # Apply steering exactly like the real emotion test
    if 'valence_128d' in signals:
        valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
        print(f"   Applying valence steering signal, shape: {valence_signal.shape}")
        model.add_steering_signal('affective_valence_128d', valence_signal, strength=5.0)
    
    if 'arousal_128d' in signals:
        arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
        print(f"   Applying arousal steering signal, shape: {arousal_signal.shape}")
        model.add_steering_signal('affective_arousal_128d', arousal_signal, strength=5.0)
    
    model.lrm.enable()
    
    # Check if signals were stored
    total_signals = 0
    for lrm_module_name, lrm_module in model.lrm.named_children():
        signals_in_module = len(lrm_module.mod_inputs)
        total_signals += signals_in_module
        print(f"   {lrm_module_name}: {signals_in_module} steering signals")
    
    print(f"   Total steering signals stored: {total_signals}")
    
    # Forward pass with first_pass_steering=False (default)
    with torch.no_grad():
        steering_output = model(sample_tensor, forward_passes=2)
    
    steering_val = steering_output['valence'].item()
    steering_aro = steering_output['arousal'].item()
    
    val_change = steering_val - baseline_val
    aro_change = steering_aro - baseline_aro
    
    print(f"   With steering: Valence={steering_val:.6f}, Arousal={steering_aro:.6f}")
    print(f"   Changes: Valence Δ={val_change:+.6f}, Arousal Δ={aro_change:+.6f}")
    
    # Test 3: Check ModBlock parameters
    print("\n🔍 Test 3: ModBlock Parameter Check")
    for lrm_module_name, lrm_module in model.lrm.named_children():
        for mod_name, mod_module in lrm_module.named_children():
            if hasattr(mod_module, 'neg_scale') and hasattr(mod_module, 'pos_scale'):
                neg_val = mod_module.neg_scale.data.item()
                pos_val = mod_module.pos_scale.data.item()
                print(f"   {mod_name}: neg_scale={neg_val:.4f}, pos_scale={pos_val:.4f}")
                
                # Check if we have original values
                has_orig = hasattr(mod_module, 'neg_scale_orig') and hasattr(mod_module, 'pos_scale_orig')
                print(f"     Has original scales: {has_orig}")
    
    # Test 4: Try with first_pass_steering=True
    print("\n🔍 Test 4: With first_pass_steering=True")
    model.lrm.clear_stored_activations()
    
    # Apply steering again
    if 'valence_128d' in signals:
        valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
        model.add_steering_signal('affective_valence_128d', valence_signal, strength=5.0)
    
    if 'arousal_128d' in signals:
        arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
        model.add_steering_signal('affective_arousal_128d', arousal_signal, strength=5.0)
    
    model.lrm.enable()
    
    # Forward pass with first_pass_steering=True
    with torch.no_grad():
        steering_output_first = model(sample_tensor, forward_passes=2, first_pass_steering=True)
    
    steering_val_first = steering_output_first['valence'].item()
    steering_aro_first = steering_output_first['arousal'].item()
    
    val_change_first = steering_val_first - baseline_val
    aro_change_first = steering_aro_first - baseline_aro
    
    print(f"   With first_pass_steering=True: Valence={steering_val_first:.6f}, Arousal={steering_aro_first:.6f}")
    print(f"   Changes: Valence Δ={val_change_first:+.6f}, Arousal Δ={aro_change_first:+.6f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if abs(val_change) > 0.001 or abs(aro_change) > 0.001:
        print("✅ SUCCESS: Steering working with first_pass_steering=False")
    elif abs(val_change_first) > 0.001 or abs(aro_change_first) > 0.001:
        print("⚠️ PARTIAL: Steering only works with first_pass_steering=True")
        print("   This suggests the issue is with second-pass modulation")
    else:
        print("❌ FAILED: Steering not working in either configuration")
        print("   This suggests a fundamental issue with the steering mechanism")

if __name__ == "__main__":
    test_steering_isolation() 