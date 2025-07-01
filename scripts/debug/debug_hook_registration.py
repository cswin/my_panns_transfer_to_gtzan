#!/usr/bin/env python3
"""
Debug hook registration to understand why forward hooks aren't triggered.
"""

import os
import sys
import torch
import numpy as np
import json

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def debug_hook_registration():
    """Debug hook registration and triggering."""
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
    
    print(f"\n🔍 HOOK REGISTRATION ANALYSIS:")
    print(f"Number of target hooks: {len(model.lrm.targ_hooks)}")
    
    # Check what modules the hooks are registered on
    for i, hook_item in enumerate(model.lrm.targ_hooks):
        if isinstance(hook_item, tuple):
            hook, hook_id = hook_item
            print(f"  Hook {i+1}: {hook_id}")
        else:
            print(f"  Hook {i+1}: {type(hook_item)}")
    
    # Let's manually check if the target modules exist
    print(f"\n🔍 TARGET MODULE VERIFICATION:")
    target_layers = [
        'visual_system.base.conv_block4',
        'visual_system.base.conv_block3', 
        'visual_system.base.conv_block2',
        'visual_system.base.conv_block1'
    ]
    
    model_layers = dict(model.named_modules())
    for target_name in target_layers:
        if target_name in model_layers:
            module = model_layers[target_name]
            print(f"  ✅ {target_name}: {type(module)}")
        else:
            print(f"  ❌ {target_name}: NOT FOUND")
    
    # Create a simple hook to test if ANY hooks are working
    def simple_test_hook(module, input, output):
        print(f"🎯 HOOK TRIGGERED: {type(module).__name__}")
        return output
    
    # Register test hooks on target layers
    test_hooks = []
    for target_name in target_layers:
        if target_name in model_layers:
            module = model_layers[target_name]
            hook = module.register_forward_hook(simple_test_hook)
            test_hooks.append(hook)
            print(f"  📌 Registered test hook on {target_name}")
    
    # Test with a sample
    print(f"\n🔧 TESTING HOOK TRIGGERING:")
    sample_tensor = torch.randn(1, 1024, 64).to(device)
    
    print("Running forward pass...")
    with torch.no_grad():
        output = model(sample_tensor, forward_passes=1)  # Single pass first
    
    print(f"Forward pass completed. Output valence: {output['valence'].cpu().item():.6f}")
    
    # Clean up test hooks
    for hook in test_hooks:
        hook.remove()
    
    # Now let's test the actual LRM hooks
    print(f"\n🔧 TESTING LRM HOOKS:")
    
    # Load steering signals
    with open('tmp/steering_signals_by_category.json', 'r') as f:
        signals_9bin = json.load(f)
    
    # Add steering signals
    model.clear_feedback_state()
    
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
    
    # Check LRM state
    print(f"LRM enabled: {not model.lrm.disable_modulation_during_inference}")
    print(f"Mod inputs count: {len(model.lrm.mod_inputs)}")
    
    # Create a more detailed debug hook
    def detailed_debug_hook(module, input, output, target_name):
        print(f"\n🎯 DETAILED HOOK for {target_name}:")
        print(f"  Module type: {type(module).__name__}")
        print(f"  Output shape: {output.shape}")
        print(f"  LRM enabled: {not model.lrm.disable_modulation_during_inference}")
        print(f"  Mod inputs available: {len(model.lrm.mod_inputs)}")
        
        # Check what modulations would be applied
        target_key = target_name.replace('.', '_')
        for mod_name, mod_module in model.lrm.named_children():
            if target_key in mod_name and mod_name in model.lrm.mod_inputs:
                print(f"  📊 Would apply: {mod_name}")
        
        return output
    
    # Replace LRM hooks temporarily
    print(f"\n🔧 REPLACING LRM HOOKS WITH DEBUG HOOKS:")
    
    # Remove existing hooks
    for hook_item in model.lrm.targ_hooks:
        if isinstance(hook_item, tuple):
            hook_item[0].remove()
        else:
            hook_item.remove()
    model.lrm.targ_hooks.clear()
    
    # Register new debug hooks
    from functools import partial
    for conn in model.lrm.mod_connections:
        target_name = conn['target']
        if target_name in model_layers:
            target_module = model_layers[target_name]
            hook = target_module.register_forward_hook(
                partial(detailed_debug_hook, target_name=target_name)
            )
            model.lrm.targ_hooks.append((hook, f"debug_{target_name}"))
            print(f"  📌 Registered debug hook on {target_name}")
    
    # Run forward pass with debug hooks
    model.lrm.enable()
    print(f"\n🔧 RUNNING FORWARD PASS WITH DEBUG HOOKS:")
    with torch.no_grad():
        output = model(sample_tensor, forward_passes=2)
    
    print(f"\n🎯 FINAL RESULT:")
    print(f"  Valence: {output['valence'].cpu().item():.6f}")
    print(f"  Arousal: {output['arousal'].cpu().item():.6f}")

if __name__ == "__main__":
    debug_hook_registration() 