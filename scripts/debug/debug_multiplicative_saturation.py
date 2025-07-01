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

def test_multiplicative_saturation():
    """Test the multiplicative modulation formula to understand saturation."""
    
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
    
    # Hook to capture modulation values
    modulation_data = {}
    
    def capture_modulation_hook(module, input, output):
        """Capture pre/post modulation values and the modulation itself."""
        target_name = None
        for name, mod in model.lrm.named_children():
            if mod is module:
                target_name = name
                break
        
        if target_name and hasattr(module, 'pre_mod_output') and hasattr(module, 'total_mod'):
            pre_mod = module.pre_mod_output
            total_mod = module.total_mod
            post_mod = module.post_mod_output
            
            if pre_mod is not None and total_mod is not None and post_mod is not None:
                modulation_data[target_name] = {
                    'pre_mod_stats': {
                        'mean': pre_mod.mean().item(),
                        'std': pre_mod.std().item(),
                        'min': pre_mod.min().item(),
                        'max': pre_mod.max().item(),
                    },
                    'total_mod_stats': {
                        'mean': total_mod.mean().item(),
                        'std': total_mod.std().item(),
                        'min': total_mod.min().item(),
                        'max': total_mod.max().item(),
                    },
                    'post_mod_stats': {
                        'mean': post_mod.mean().item(),
                        'std': post_mod.std().item(),
                        'min': post_mod.min().item(),
                        'max': post_mod.max().item(),
                    }
                }
                
                # Calculate the multiplicative effect
                # Formula: post = pre + pre * total_mod = pre * (1 + total_mod)
                multiplier = 1 + total_mod
                modulation_data[target_name]['multiplier_stats'] = {
                    'mean': multiplier.mean().item(),
                    'std': multiplier.std().item(),
                    'min': multiplier.min().item(),
                    'max': multiplier.max().item(),
                }
    
    # Register hooks on LRM modules
    hooks = []
    for name, lrm_module in model.lrm.named_children():
        hooks.append(lrm_module.register_forward_hook(capture_modulation_hook))
    
    print("\n" + "="*80)
    print("MULTIPLICATIVE MODULATION ANALYSIS")
    print("="*80)
    
    # Test different strengths
    strengths = [0.1, 1.0, 10.0, 100.0]
    
    for strength in strengths:
        print(f"\n🔍 Testing strength: {strength}")
        
        # Clear previous state
        model.lrm.clear_stored_activations()
        modulation_data.clear()
        
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
        
        print(f"  Final predictions: Valence={output['valence'].item():.6f}, Arousal={output['arousal'].item():.6f}")
        
        # Analyze modulation data
        for target_name, data in modulation_data.items():
            print(f"\n  📊 {target_name}:")
            pre = data['pre_mod_stats']
            mod = data['total_mod_stats']
            post = data['post_mod_stats']
            mult = data['multiplier_stats']
            
            print(f"    Pre-modulation:  mean={pre['mean']:8.6f}, std={pre['std']:8.6f}, range=[{pre['min']:8.6f}, {pre['max']:8.6f}]")
            print(f"    Total modulation: mean={mod['mean']:8.6f}, std={mod['std']:8.6f}, range=[{mod['min']:8.6f}, {mod['max']:8.6f}]")
            print(f"    Multiplier (1+mod): mean={mult['mean']:8.6f}, std={mult['std']:8.6f}, range=[{mult['min']:8.6f}, {mult['max']:8.6f}]")
            print(f"    Post-modulation: mean={post['mean']:8.6f}, std={post['std']:8.6f}, range=[{post['min']:8.6f}, {post['max']:8.6f}]")
            
            # Check for saturation indicators
            if mult['max'] > 10.0 or mult['min'] < -5.0:
                print(f"    ⚠️ EXTREME MULTIPLIERS: range=[{mult['min']:.3f}, {mult['max']:.3f}]")
            
            if abs(mod['mean']) > 1.0:
                print(f"    ⚠️ LARGE MODULATION: mean={mod['mean']:.3f}")
            
            # Check if ReLU is clipping
            if post['min'] == 0.0 and pre['min'] < 0.0:
                print(f"    ⚠️ ReLU CLIPPING: pre_min={pre['min']:.3f} -> post_min={post['min']:.3f}")
        
        model.lrm.reset_modulation_strength()
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    print("\n" + "="*80)
    print("SATURATION ANALYSIS SUMMARY")
    print("="*80)
    
    # Test the multiplicative formula directly with synthetic data
    print("\n🧪 SYNTHETIC MULTIPLICATIVE TEST:")
    
    # Simulate typical CNN feature values
    base_features = torch.randn(1, 128, 8, 8) * 0.5  # Typical CNN feature magnitude
    
    # Simulate modulation values at different strengths
    mod_strengths = [0.01, 0.1, 1.0, 10.0]
    
    for mod_strength in mod_strengths:
        # Simulate modulation similar to what ModBlocks produce
        modulation = torch.randn_like(base_features) * mod_strength
        
        # Apply multiplicative modulation
        result = base_features + base_features * modulation
        result_relu = F.relu(result)
        
        print(f"  Mod strength {mod_strength:4.2f}: "
              f"base_mean={base_features.mean():.6f}, "
              f"mod_mean={modulation.mean():.6f}, "
              f"result_mean={result.mean():.6f}, "
              f"relu_mean={result_relu.mean():.6f}")
        
        # Check for saturation
        if result_relu.mean() / base_features.mean() > 5.0:
            print(f"    ⚠️ AMPLIFICATION SATURATION: {result_relu.mean() / base_features.mean():.2f}x amplification")

if __name__ == "__main__":
    test_multiplicative_saturation() 