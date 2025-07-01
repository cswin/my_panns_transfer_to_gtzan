#!/usr/bin/env python3
"""
Debug script to examine ModBlock parameter values and identify saturation sources.
"""

import sys
import os
sys.path.append('src')

import torch
import numpy as np
import json

from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def debug_modblock_parameters():
    """Debug ModBlock parameters at different steering strengths."""
    print("=== Debugging ModBlock Parameters ===")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, 
        mel_bins=64, fmin=50, fmax=14000, forward_passes=2
    )
    model.load_from_pretrain('/DATA/pliu/EmotionData/Cnn6_mAP=0.343.pth')
    model = model.to(device)
    model.eval()
    
    # Load steering signals
    with open('steering_signals_25bin/steering_signals_25bin.json', 'r') as f:
        steering_data = json.load(f)
    
    # Get a test steering signal
    category = 'positive_strong'
    if category not in steering_data:
        category = list(steering_data.keys())[0]  # Use first available
    
    valence_signal = torch.tensor(steering_data[category]['valence_128d'], dtype=torch.float32).to(device)
    arousal_signal = torch.tensor(steering_data[category]['arousal_128d'], dtype=torch.float32).to(device)
    
    print(f"Using steering signals from category: {category}")
    print(f"Valence signal stats: mean={valence_signal.mean():.6f}, std={valence_signal.std():.6f}")
    print(f"Arousal signal stats: mean={arousal_signal.mean():.6f}, std={arousal_signal.std():.6f}")
    
    # Test different strength values
    strength_values = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 1000.0]
    
    print(f"\n{'='*80}")
    print(f"MODBLOCK PARAMETER ANALYSIS")
    print(f"{'='*80}")
    
    for strength in strength_values:
        print(f"\n🎯 Testing strength: {strength}")
        print("-" * 60)
        
        # Clear any previous state
        model.clear_feedback_state()
        model.lrm.clear_stored_activations()
        
        # Store original parameter values
        original_params = {}
        for lrm_module_name, lrm_module in model.lrm.named_children():
            for mod_name, mod_module in lrm_module.named_children():
                if hasattr(mod_module, 'neg_scale') and hasattr(mod_module, 'pos_scale'):
                    original_params[f"{lrm_module_name}.{mod_name}"] = {
                        'neg_scale': mod_module.neg_scale.item(),
                        'pos_scale': mod_module.pos_scale.item()
                    }
        
        # Apply steering signals
        steering_signals_list = [
            {'source': 'affective_valence_128d', 'activation': valence_signal, 'strength': strength, 'alpha': 1.0},
            {'source': 'affective_arousal_128d', 'activation': arousal_signal, 'strength': strength, 'alpha': 1.0}
        ]
        
        # Add steering signals (this should modify the parameters)
        for signal_dict in steering_signals_list:
            model.add_steering_signal(**signal_dict)
        
        # Check parameter values after applying steering
        print(f"Parameter changes after applying strength {strength}:")
        print(f"{'Module':<50} {'Original neg_scale':<15} {'New neg_scale':<15} {'Original pos_scale':<15} {'New pos_scale':<15}")
        print("-" * 120)
        
        modified_count = 0
        for lrm_module_name, lrm_module in model.lrm.named_children():
            for mod_name, mod_module in lrm_module.named_children():
                if hasattr(mod_module, 'neg_scale') and hasattr(mod_module, 'pos_scale'):
                    module_key = f"{lrm_module_name}.{mod_name}"
                    if module_key in original_params:
                        orig_neg = original_params[module_key]['neg_scale']
                        orig_pos = original_params[module_key]['pos_scale']
                        new_neg = mod_module.neg_scale.item()
                        new_pos = mod_module.pos_scale.item()
                        
                        # Check if parameters were modified
                        if abs(new_neg - orig_neg) > 1e-6 or abs(new_pos - orig_pos) > 1e-6:
                            modified_count += 1
                            print(f"{module_key:<50} {orig_neg:<15.6f} {new_neg:<15.6f} {orig_pos:<15.6f} {new_pos:<15.6f}")
        
        if modified_count == 0:
            print("❌ No parameters were modified!")
        else:
            print(f"✅ {modified_count} ModBlocks had parameters modified")
        
        # Test actual forward pass with these parameters
        print(f"\nTesting forward pass with strength {strength}:")
        
        # Create dummy input
        dummy_input = torch.randn(1, 1024, 64).to(device)
        
        # Baseline prediction
        model.clear_feedback_state()
        model.lrm.clear_stored_activations()
        with torch.no_grad():
            baseline_output = model(dummy_input, forward_passes=2)
            baseline_valence = baseline_output['valence'].item()
            baseline_arousal = baseline_output['arousal'].item()
        
        # Steered prediction
        model.clear_feedback_state()
        model.lrm.clear_stored_activations()
        for signal_dict in steering_signals_list:
            model.add_steering_signal(**signal_dict)
        
        with torch.no_grad():
            model.lrm.enable()
            steered_output = model(dummy_input, forward_passes=2, steering_signals=steering_signals_list, first_pass_steering=False)
            steered_valence = steered_output['valence'].item()
            steered_arousal = steered_output['arousal'].item()
        
        valence_change = steered_valence - baseline_valence
        arousal_change = steered_arousal - baseline_arousal
        
        print(f"  Baseline: V={baseline_valence:.6f}, A={baseline_arousal:.6f}")
        print(f"  Steered:  V={steered_valence:.6f}, A={steered_arousal:.6f}")
        print(f"  Changes:  ΔV={valence_change:+.6f}, ΔA={arousal_change:+.6f}")
        
        # Reset parameters for next iteration
        model.reset_modulation_strengths()
    
    print(f"\n{'='*80}")
    print(f"DETAILED MODBLOCK STRUCTURE ANALYSIS")
    print(f"{'='*80}")
    
    # Analyze the ModBlock structure in detail
    print("\nLRM Module Structure:")
    for lrm_module_name, lrm_module in model.lrm.named_children():
        print(f"\n📊 LRM Module: {lrm_module_name}")
        for mod_name, mod_module in lrm_module.named_children():
            print(f"  ModBlock: {mod_name}")
            if hasattr(mod_module, 'neg_scale'):
                print(f"    neg_scale: {mod_module.neg_scale.item():.6f}")
            if hasattr(mod_module, 'pos_scale'):
                print(f"    pos_scale: {mod_module.pos_scale.item():.6f}")
            if hasattr(mod_module, 'rescale'):
                print(f"    rescale: {type(mod_module.rescale).__name__}")
                if hasattr(mod_module.rescale, 'squash'):
                    print(f"      squash: {type(mod_module.rescale.squash).__name__}")
                    print(f"      squash mode: {mod_module.rescale.squash.mode}")
            if hasattr(mod_module, 'modulation'):
                conv = mod_module.modulation
                print(f"    modulation: Conv2d({conv.in_channels}, {conv.out_channels})")
                if hasattr(conv, 'weight'):
                    weight_stats = conv.weight.data
                    print(f"      weight stats: mean={weight_stats.mean():.6f}, std={weight_stats.std():.6f}")
                if hasattr(conv, 'bias') and conv.bias is not None:
                    bias_stats = conv.bias.data
                    print(f"      bias stats: mean={bias_stats.mean():.6f}, std={bias_stats.std():.6f}")

def test_modblock_processing_pipeline():
    """Test the ModBlock processing pipeline step by step."""
    print(f"\n{'='*80}")
    print(f"MODBLOCK PROCESSING PIPELINE TEST")
    print(f"{'='*80}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, 
        mel_bins=64, fmin=50, fmax=14000, forward_passes=2
    )
    model.load_from_pretrain('/DATA/pliu/EmotionData/Cnn6_mAP=0.343.pth')
    model = model.to(device)
    
    # Get a ModBlock for testing
    test_modblock = None
    test_modblock_name = None
    for lrm_module_name, lrm_module in model.lrm.named_children():
        for mod_name, mod_module in lrm_module.named_children():
            if hasattr(mod_module, 'rescale') and hasattr(mod_module, 'neg_scale'):
                test_modblock = mod_module
                test_modblock_name = f"{lrm_module_name}.{mod_name}"
                break
        if test_modblock is not None:
            break
    
    if test_modblock is None:
        print("❌ No suitable ModBlock found for testing!")
        return
    
    print(f"Testing ModBlock: {test_modblock_name}")
    
    # Create test input (128D signal)
    test_input = torch.randn(1, 128, 1, 1).to(device) * 2.0  # Scale to make it more visible
    target_size = (7, 7)  # Typical conv layer size
    
    print(f"Input stats: mean={test_input.mean():.6f}, std={test_input.std():.6f}, range=[{test_input.min():.6f}, {test_input.max():.6f}]")
    
    # Test different strength values
    strengths = [0.1, 1.0, 10.0, 100.0]
    
    for strength in strengths:
        print(f"\n🔍 Testing strength: {strength}")
        
        # Reset parameters
        test_modblock.neg_scale.data.fill_(1.0)
        test_modblock.pos_scale.data.fill_(1.0)
        
        # Apply strength scaling
        test_modblock.neg_scale.data *= strength
        test_modblock.pos_scale.data *= strength
        
        print(f"  Modified scales: neg_scale={test_modblock.neg_scale.item():.6f}, pos_scale={test_modblock.pos_scale.item():.6f}")
        
        with torch.no_grad():
            # Step 1: Rescale (normalize, squash, resize)
            x_after_rescale = test_modblock.rescale(test_input, target_size)
            print(f"  After rescale: mean={x_after_rescale.mean():.6f}, std={x_after_rescale.std():.6f}, range=[{x_after_rescale.min():.6f}, {x_after_rescale.max():.6f}]")
            
            # Step 2: Apply learnable scales
            neg_mask, pos_mask = x_after_rescale < 0, x_after_rescale >= 0
            x_after_scales = x_after_rescale * (neg_mask.float() * test_modblock.neg_scale + pos_mask.float() * test_modblock.pos_scale)
            print(f"  After scales: mean={x_after_scales.mean():.6f}, std={x_after_scales.std():.6f}, range=[{x_after_scales.min():.6f}, {x_after_scales.max():.6f}]")
            
            # Step 3: Conv1x1 modulation
            x_final = test_modblock.modulation(x_after_scales)
            print(f"  After conv1x1: mean={x_final.mean():.6f}, std={x_final.std():.6f}, range=[{x_final.min():.6f}, {x_final.max():.6f}]")
            
            # Full ModBlock forward
            x_full = test_modblock(test_input, target_size)
            print(f"  Full forward: mean={x_full.mean():.6f}, std={x_full.std():.6f}, range=[{x_full.min():.6f}, {x_full.max():.6f}]")

if __name__ == "__main__":
    debug_modblock_parameters()
    test_modblock_processing_pipeline() 