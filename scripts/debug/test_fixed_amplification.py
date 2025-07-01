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

def test_fixed_amplification():
    """Test the fixed amplification approach."""
    
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
    print("FIXED AMPLIFICATION TEST (3x scaling + clamping)")
    print("="*80)
    
    # Get baseline
    model.lrm.clear_stored_activations()
    model.lrm.disable()
    
    with torch.no_grad():
        baseline = model(sample_tensor, forward_passes=2)
    
    baseline_val = baseline['valence'].item()
    baseline_aro = baseline['arousal'].item()
    print(f"\n📊 Baseline: Valence={baseline_val:.6f}, Arousal={baseline_aro:.6f}")
    
    # Test different strengths
    strengths = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    print(f"\n{'Strength':>8} {'Valence':>10} {'Arousal':>10} {'Val Δ':>10} {'Aro Δ':>10} {'Total Δ':>10}")
    print("-" * 68)
    
    results = []
    for strength in strengths:
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
            'val_change': val_change,
            'aro_change': aro_change,
            'total_change': total_change
        })
        
        print(f"{strength:>8.1f} {output['valence'].item():>10.6f} {output['arousal'].item():>10.6f} "
              f"{val_change:>+10.6f} {aro_change:>+10.6f} {total_change:>10.6f}")
        
        model.lrm.reset_modulation_strength()
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Check scaling behavior
    print(f"\n📈 Scaling Analysis:")
    for i in range(1, len(results)):
        prev_change = results[i-1]['total_change']
        curr_change = results[i]['total_change']
        ratio = curr_change / prev_change if prev_change > 0 else 0
        
        print(f"   Strength {results[i-1]['strength']} → {results[i]['strength']}: "
              f"{prev_change:.6f} → {curr_change:.6f} (ratio: {ratio:.2f})")
    
    # Summary
    max_effect = max(results, key=lambda x: x['total_change'])
    min_effect = min(results, key=lambda x: x['total_change'])
    
    print(f"\n📊 Summary:")
    print(f"   Minimum effect: {min_effect['total_change']:.6f} at strength {min_effect['strength']}")
    print(f"   Maximum effect: {max_effect['total_change']:.6f} at strength {max_effect['strength']}")
    print(f"   Dynamic range: {max_effect['total_change'] / min_effect['total_change']:.1f}x")
    
    # Check if we have proper scaling
    early_effects = [r['total_change'] for r in results[:3]]
    late_effects = [r['total_change'] for r in results[-3:]]
    
    if max(early_effects) < max(late_effects):
        print("   ✅ Good: Effects increase with strength")
    else:
        print("   ⚠️ Issue: Effects don't consistently increase with strength")

if __name__ == "__main__":
    test_fixed_amplification() 