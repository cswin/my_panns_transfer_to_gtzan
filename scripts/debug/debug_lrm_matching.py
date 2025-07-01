#!/usr/bin/env python3
"""
Debug LRM matching logic to identify why signals converge.
"""

import os
import sys
import torch
import numpy as np
import json

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def debug_lrm_matching():
    """Debug the LRM matching logic to understand signal convergence."""
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
    
    print(f"\n🔍 ANALYZING LRM STRUCTURE:")
    print(f"LRM connections: {len(model.lrm.mod_connections)}")
    for i, conn in enumerate(model.lrm.mod_connections):
        print(f"  {i+1}. {conn['source']} -> {conn['target']}")
    
    print(f"\n🔍 LRM ModBlocks:")
    for name, module in model.lrm.named_children():
        print(f"  - {name}")
    
    # Test the matching logic
    print(f"\n🔍 TESTING MATCHING LOGIC:")
    target_layers = [
        'visual_system.base.conv_block4',
        'visual_system.base.conv_block3', 
        'visual_system.base.conv_block2',
        'visual_system.base.conv_block1'
    ]
    
    for target_name in target_layers:
        print(f"\n📊 Target: {target_name}")
        target_key = target_name.replace('.', '_')
        
        matching_blocks = []
        for mod_name, mod_module in model.lrm.named_children():
            # Current matching logic
            if target_key in mod_name:
                matching_blocks.append(mod_name)
        
        print(f"  Matching ModBlocks ({len(matching_blocks)}):")
        for block in matching_blocks:
            print(f"    - {block}")
        
        if len(matching_blocks) > 1:
            print(f"  ⚠️  MULTIPLE MATCHES - This causes signal mixing!")
    
    # Test with actual steering signals
    print(f"\n🔍 TESTING WITH STEERING SIGNALS:")
    
    # Load steering signals
    with open('tmp/steering_signals_by_category.json', 'r') as f:
        signals_9bin = json.load(f)
    
    # Add different steering signals
    model.clear_feedback_state()
    
    # Add valence signal
    if 'negative_strong' in signals_9bin:
        signals = signals_9bin['negative_strong']
        if 'valence_128d' in signals:
            valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
            model.add_steering_signal(
                source='affective_valence_128d',
                activation=valence_signal,
                strength=5.0,
                alpha=1.0
            )
            print(f"  ✅ Added valence steering signal")
    
    # Add arousal signal  
    if 'negative_strong' in signals_9bin:
        signals = signals_9bin['negative_strong']
        if 'arousal_128d' in signals:
            arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
            model.add_steering_signal(
                source='affective_arousal_128d',
                activation=arousal_signal,
                strength=5.0,
                alpha=1.0
            )
            print(f"  ✅ Added arousal steering signal")
    
    # Check what's stored in mod_inputs
    print(f"\n🔍 STORED MOD_INPUTS:")
    for key, value in model.lrm.mod_inputs.items():
        print(f"  - {key}: shape={value.shape}")
    
    # Create a custom forward hook to see what happens during modulation
    def debug_hook(module, input, output, target_name):
        print(f"\n🎯 MODULATION DEBUG for {target_name}:")
        print(f"  Output shape: {output.shape}")
        
        target_key = target_name.replace('.', '_')
        total_mods = 0
        
        for mod_name, mod_module in model.lrm.named_children():
            if target_key in mod_name and mod_name in model.lrm.mod_inputs:
                total_mods += 1
                source_activation = model.lrm.mod_inputs[mod_name]
                print(f"  📊 ModBlock {mod_name}:")
                print(f"     Source shape: {source_activation.shape}")
                print(f"     Source mean: {source_activation.mean().item():.6f}")
                print(f"     Source std: {source_activation.std().item():.6f}")
        
        print(f"  Total ModBlocks applied: {total_mods}")
        return output
    
    # Replace the forward hook temporarily
    original_hook = model.lrm.forward_hook_target
    model.lrm.forward_hook_target = debug_hook
    
    # Test with a sample
    sample_tensor = torch.randn(1, 1024, 64).to(device)
    model.lrm.enable()
    
    print(f"\n🔧 RUNNING FORWARD PASS WITH DEBUG:")
    with torch.no_grad():
        output = model(sample_tensor, forward_passes=2)
    
    # Restore original hook
    model.lrm.forward_hook_target = original_hook
    
    print(f"\n🎯 FINAL OUTPUT:")
    print(f"  Valence: {output['valence'].cpu().item():.6f}")
    print(f"  Arousal: {output['arousal'].cpu().item():.6f}")

if __name__ == "__main__":
    debug_lrm_matching() 