#!/usr/bin/env python3

import sys
import os
# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import json
import numpy as np
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def test_modulation_amplification():
    """Test the modulation amplification fix."""
    
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
    available_categories = list(signals_25bin.keys())
    category = None
    for cat in available_categories:
        if 'valence_128d' in signals_25bin[cat] and 'arousal_128d' in signals_25bin[cat]:
            category = cat
            break
    
    signals = signals_25bin[category]
    print(f"🎯 Using category: {category}")
    
    # Test sample
    sample_tensor = torch.randn(1, 1024, 64).to(device)
    
    print("\n" + "="*80)
    print("MODULATION AMPLIFICATION TEST")
    print("="*80)
    
    # Get baseline (no steering)
    model.lrm.clear_stored_activations()
    model.lrm.disable()
    
    with torch.no_grad():
        baseline = model(sample_tensor, forward_passes=2)
    
    baseline_val = baseline['valence'].item()
    baseline_aro = baseline['arousal'].item()
    print(f"\n📊 Baseline (no steering):")
    print(f"   Valence: {baseline_val:.6f}")
    print(f"   Arousal: {baseline_aro:.6f}")
    
    # Test different strengths with amplification
    strengths = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    results = []
    
    for strength in strengths:
        print(f"\n🔍 Testing strength: {strength}")
        
        # Clear previous state
        model.lrm.clear_stored_activations()
        
        # Apply steering signals
        if 'valence_128d' in signals:
            valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
            model.add_steering_signal('affective_valence_128d', valence_signal, strength=strength)
        
        if 'arousal_128d' in signals:
            arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
            model.add_steering_signal('affective_arousal_128d', arousal_signal, strength=strength)
        
        model.lrm.enable()
        
        # Forward pass
        with torch.no_grad():
            output = model(sample_tensor, forward_passes=2)
        
        val_change = output['valence'].item() - baseline_val
        aro_change = output['arousal'].item() - baseline_aro
        total_change = abs(val_change) + abs(aro_change)
        
        results.append({
            'strength': strength,
            'valence': output['valence'].item(),
            'arousal': output['arousal'].item(),
            'val_change': val_change,
            'aro_change': aro_change,
            'total_change': total_change
        })
        
        print(f"   Valence: {output['valence'].item():.6f} (Δ: {val_change:+.6f})")
        print(f"   Arousal: {output['arousal'].item():.6f} (Δ: {aro_change:+.6f})")
        print(f"   Total change: {total_change:.6f}")
        
        model.lrm.reset_modulation_strength()
    
    print("\n" + "="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    # Analyze the scaling behavior
    print(f"\n📈 Steering Effect Scaling:")
    print(f"{'Strength':>8} {'Val Δ':>10} {'Aro Δ':>10} {'Total Δ':>10} {'Scale Ratio':>12}")
    print("-" * 54)
    
    prev_total = 0
    for i, result in enumerate(results):
        scale_ratio = result['total_change'] / results[0]['total_change'] if results[0]['total_change'] > 0 else 0
        print(f"{result['strength']:>8.1f} {result['val_change']:>+10.6f} {result['aro_change']:>+10.6f} "
              f"{result['total_change']:>10.6f} {scale_ratio:>12.2f}x")
        prev_total = result['total_change']
    
    # Check for plateau behavior
    print(f"\n🔍 Plateau Analysis:")
    change_ratios = []
    for i in range(1, len(results)):
        if results[i-1]['total_change'] > 0:
            ratio = results[i]['total_change'] / results[i-1]['total_change']
            change_ratios.append(ratio)
            if ratio < 1.2:  # Less than 20% increase
                print(f"   Strength {results[i-1]['strength']} → {results[i]['strength']}: "
                      f"Ratio = {ratio:.2f} (plateau detected)")
    
    # Summary
    max_effect = max(results, key=lambda x: x['total_change'])
    print(f"\n📊 Summary:")
    print(f"   Baseline effects: {results[0]['total_change']:.6f}")
    print(f"   Maximum effects: {max_effect['total_change']:.6f} at strength {max_effect['strength']}")
    print(f"   Amplification: {max_effect['total_change'] / results[0]['total_change']:.1f}x")
    
    # Check if we achieved better scaling
    if len(change_ratios) > 2:
        avg_early_ratio = np.mean(change_ratios[:3])  # First 3 ratios
        avg_late_ratio = np.mean(change_ratios[-3:])  # Last 3 ratios
        
        print(f"   Early scaling (avg): {avg_early_ratio:.2f}x per step")
        print(f"   Late scaling (avg): {avg_late_ratio:.2f}x per step")
        
        if avg_early_ratio > 1.5 and avg_late_ratio < 1.2:
            print("   ✅ Good scaling: Strong early effects, plateau at high strengths")
        elif avg_early_ratio < 1.2:
            print("   ⚠️ Weak scaling: Effects plateau too early")
        else:
            print("   ⚠️ No plateau: May need stronger clamping")

if __name__ == "__main__":
    test_modulation_amplification() 