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

def test_adaptive_amplification():
    """Test the adaptive amplification fix with debug output."""
    
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
    
    # Hook to capture scale factors and modulation stats
    debug_info = {}
    
    def debug_hook(module, input, output):
        """Capture modulation debug info."""
        target_name = None
        for name, mod in model.lrm.named_children():
            if mod is module:
                target_name = name
                break
        
        if target_name and hasattr(module, 'total_mod') and module.total_mod is not None:
            # Get the original total_mod (before our modifications)
            original_mod = module.total_mod
            
            # Calculate what our adaptive scaling would do
            mod_magnitude = torch.abs(original_mod).mean().item()
            
            if mod_magnitude < 0.01:
                scale_factor = 10.0
            elif mod_magnitude < 0.1:
                scale_factor = 5.0
            elif mod_magnitude < 0.5:
                scale_factor = 2.0
            else:
                scale_factor = 1.0
            
            debug_info[target_name] = {
                'original_magnitude': mod_magnitude,
                'scale_factor': scale_factor,
                'original_range': [original_mod.min().item(), original_mod.max().item()],
                'scaled_magnitude': mod_magnitude * scale_factor
            }
    
    # Register debug hooks
    hooks = []
    for name, lrm_module in model.lrm.named_children():
        hooks.append(lrm_module.register_forward_hook(debug_hook))
    
    print("\n" + "="*80)
    print("ADAPTIVE AMPLIFICATION TEST")
    print("="*80)
    
    # Get baseline
    model.lrm.clear_stored_activations()
    model.lrm.disable()
    
    with torch.no_grad():
        baseline = model(sample_tensor, forward_passes=2)
    
    baseline_val = baseline['valence'].item()
    baseline_aro = baseline['arousal'].item()
    print(f"\n📊 Baseline (no steering): Valence={baseline_val:.6f}, Arousal={baseline_aro:.6f}")
    
    # Test different strengths
    strengths = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for strength in strengths:
        print(f"\n🔍 Testing strength: {strength}")
        
        # Clear previous state
        model.lrm.clear_stored_activations()
        debug_info.clear()
        
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
        
        print(f"   Results: Valence Δ={val_change:+.6f}, Arousal Δ={aro_change:+.6f}, Total={total_change:.6f}")
        
        # Show debug info
        print(f"   Debug info:")
        for target_name, info in debug_info.items():
            print(f"     {target_name}:")
            print(f"       Original magnitude: {info['original_magnitude']:.6f}")
            print(f"       Scale factor: {info['scale_factor']:.1f}x")
            print(f"       Scaled magnitude: {info['scaled_magnitude']:.6f}")
            print(f"       Original range: [{info['original_range'][0]:.6f}, {info['original_range'][1]:.6f}]")
        
        model.lrm.reset_modulation_strength()
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("The adaptive scaling should:")
    print("- Apply 10x scaling for very small modulations (< 0.01)")
    print("- Apply 5x scaling for small modulations (< 0.1)")
    print("- Apply 2x scaling for medium modulations (< 0.5)")
    print("- Apply 1x scaling for large modulations (≥ 0.5)")
    print("- This should create smoother scaling across strengths")

if __name__ == "__main__":
    test_adaptive_amplification() 