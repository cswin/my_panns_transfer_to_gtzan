#!/usr/bin/env python3

import sys
import os
# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import torch.nn.functional as F
import json
import numpy as np
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def debug_modulation_saturation():
    """Debug where saturation occurs in the modulation pipeline."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, forward_passes=2
    ).to(device)
    
    # Load pretrained weights (optional for saturation analysis)
    checkpoint_path = '/DATA/pliu/EmotionData/Cnn6_mAP=0.343.pth'
    if os.path.exists(checkpoint_path):
        model.load_from_pretrain(checkpoint_path)
        print("✅ Loaded pretrained weights")
    else:
        print("⚠️ Pretrained weights not found, using random initialization")
        print("   (This is fine for analyzing saturation behavior)")
    
    # Load steering signals
    steering_signals_path = './steering_signals_25bin/steering_signals_25bin.json'
    if not os.path.exists(steering_signals_path):
        steering_signals_path = './tmp/25bin_steering_signals/steering_signals_25bin.json'
    
    with open(steering_signals_path, 'r') as f:
        signals_25bin = json.load(f)
    print(f"✅ Loaded steering signals from {steering_signals_path}")
    
    # Show available categories
    available_categories = list(signals_25bin.keys())
    print(f"📋 Available categories ({len(available_categories)}): {available_categories[:5]}...")
    
    # Test sample
    sample_tensor = torch.randn(1, 1024, 64).to(device)
    # Use the first available category that has both valence and arousal signals
    category = None
    for cat in available_categories:
        if 'valence_128d' in signals_25bin[cat] and 'arousal_128d' in signals_25bin[cat]:
            category = cat
            break
    
    if category is None:
        print("❌ No category found with both valence and arousal signals")
        return
    
    print(f"🎯 Using category: {category}")
    
    signals = signals_25bin[category]
    
    # Hook to capture intermediate feature statistics
    feature_stats = {}
    
    def make_feature_hook(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                feature_stats[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'min': output.min().item(),
                    'max': output.max().item(),
                    'shape': list(output.shape)
                }
        return hook
    
    # Register hooks on key layers
    hooks = []
    target_layers = [
        'visual_system.base.conv_block1',
        'visual_system.base.conv_block2', 
        'visual_system.base.conv_block3',
        'visual_system.base.conv_block4',
        'visual_system.base.fc1',
        'affective_valence',
        'affective_arousal'
    ]
    
    for layer_name in target_layers:
        try:
            layer = model
            for part in layer_name.split('.'):
                layer = getattr(layer, part)
            hooks.append(layer.register_forward_hook(make_feature_hook(layer_name)))
        except AttributeError:
            print(f"⚠️ Layer {layer_name} not found")
    
    print("\n" + "="*80)
    print("MODULATION SATURATION ANALYSIS")
    print("="*80)
    
    # Test different strengths
    strengths = [0.0, 0.1, 1.0, 10.0, 100.0]
    
    for strength in strengths:
        print(f"\n🔍 Testing strength: {strength}")
        
        # Clear previous state
        model.lrm.clear_stored_activations()
        feature_stats.clear()
        
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
        
        # Print results
        print(f"  Final predictions: Valence={output['valence'].item():.6f}, Arousal={output['arousal'].item():.6f}")
        
        # Print feature statistics
        for layer_name in target_layers:
            if layer_name in feature_stats:
                stats = feature_stats[layer_name]
                print(f"    {layer_name}: mean={stats['mean']:.6f}, std={stats['std']:.6f}, range=[{stats['min']:.6f}, {stats['max']:.6f}]")
        
        # Check for saturation indicators
        saturation_indicators = []
        for layer_name, stats in feature_stats.items():
            # Check for ReLU saturation (many zeros)
            if stats['min'] == 0.0 and stats['std'] < 0.1:
                saturation_indicators.append(f"{layer_name}: ReLU saturation (std={stats['std']:.6f})")
            
            # Check for extreme values
            if abs(stats['max']) > 10.0 or abs(stats['min']) > 10.0:
                saturation_indicators.append(f"{layer_name}: Extreme values (range=[{stats['min']:.3f}, {stats['max']:.3f}])")
        
        if saturation_indicators:
            print(f"  ⚠️ Saturation indicators:")
            for indicator in saturation_indicators:
                print(f"    - {indicator}")
        
        model.lrm.reset_modulation_strength()
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    print("\n" + "="*80)
    print("MODULATION EFFECT ANALYSIS")
    print("="*80)
    
    # Compare baseline vs modulated predictions
    baseline_outputs = []
    modulated_outputs = []
    
    model.lrm.clear_stored_activations()
    model.lrm.disable()
    
    # Baseline (no modulation)
    with torch.no_grad():
        baseline = model(sample_tensor, forward_passes=1)
        baseline_outputs.append(baseline)
    
    # Test with different modulation strengths
    for strength in [0.1, 1.0, 10.0]:
        model.lrm.clear_stored_activations()
        
        # Apply steering signals
        if 'valence_128d' in signals:
            valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
            model.add_steering_signal('affective_valence_128d', valence_signal, strength=strength)
        
        if 'arousal_128d' in signals:
            arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
            model.add_steering_signal('affective_arousal_128d', arousal_signal, strength=strength)
        
        model.lrm.enable()
        
        with torch.no_grad():
            modulated = model(sample_tensor, forward_passes=2)
            modulated_outputs.append((strength, modulated))
        
        model.lrm.reset_modulation_strength()
    
    # Analysis
    print(f"\nBaseline: Valence={baseline['valence'].item():.6f}, Arousal={baseline['arousal'].item():.6f}")
    
    for strength, modulated in modulated_outputs:
        val_change = modulated['valence'].item() - baseline['valence'].item()
        aro_change = modulated['arousal'].item() - baseline['arousal'].item()
        print(f"Strength {strength:4.1f}: Valence Δ={val_change:+.6f}, Arousal Δ={aro_change:+.6f}")
    
    # Check for plateau behavior
    if len(modulated_outputs) >= 2:
        changes = []
        for strength, modulated in modulated_outputs:
            val_change = abs(modulated['valence'].item() - baseline['valence'].item())
            aro_change = abs(modulated['arousal'].item() - baseline['arousal'].item())
            changes.append((strength, val_change + aro_change))
        
        # Check if changes plateau
        if len(changes) >= 3:
            change_ratios = []
            for i in range(1, len(changes)):
                if changes[i-1][1] > 0:
                    ratio = changes[i][1] / changes[i-1][1]
                    change_ratios.append(ratio)
            
            if change_ratios and max(change_ratios) < 1.5:  # Changes increase by less than 50%
                print(f"\n⚠️ PLATEAU DETECTED: Changes not scaling with strength")
                print(f"   Change ratios: {[f'{r:.2f}' for r in change_ratios]}")

if __name__ == "__main__":
    debug_modulation_saturation() 