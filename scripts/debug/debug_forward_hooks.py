#!/usr/bin/env python3
"""
Debug script to test if forward hooks are being triggered.
"""

import torch
import sys
import os

# Add src to path
sys.path.append('src')

from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
from utils.config import cnn6_config

def test_forward_hooks():
    """Test if forward hooks are being triggered."""
    print("=== Testing Forward Hooks ===")
    
    # Create model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000,
        window_size=cnn6_config['window_size'],
        hop_size=cnn6_config['hop_size'],
        mel_bins=cnn6_config['mel_bins'],
        fmin=cnn6_config['fmin'],
        fmax=cnn6_config['fmax'],
        freeze_base=True,
        forward_passes=2
    )
    
    print(f"Model created successfully")
    print(f"LRM modules: {list(model.lrm.named_children())}")
    
    # Check if hooks are registered
    print(f"\n=== Checking Hook Registration ===")
    for lrm_module_name, lrm_module in model.lrm.named_children():
        print(f"{lrm_module_name}:")
        print(f"  targ_hooks: {len(lrm_module.targ_hooks)}")
        print(f"  mod_hooks: {len(lrm_module.mod_hooks)}")
        print(f"  active_connections: {lrm_module.active_connections}")
    
    # Create dummy input
    dummy_input = torch.randn(1, 1024, 64)  # (batch, time, mel_bins)
    print(f"\n=== Testing Forward Pass ===")
    print(f"Input shape: {dummy_input.shape}")
    
    # Run forward pass
    with torch.no_grad():
        output = model(dummy_input, forward_passes=1)
        print(f"Output valence: {output['valence'].item():.4f}")
        print(f"Output arousal: {output['arousal'].item():.4f}")
    
    # Check if hooks were triggered
    print(f"\n=== Checking Hook Results ===")
    for lrm_module_name, lrm_module in model.lrm.named_children():
        print(f"{lrm_module_name}:")
        print(f"  mod_inputs keys: {list(lrm_module.mod_inputs.keys())}")
        print(f"  pre_mod_output: {lrm_module.pre_mod_output is not None}")
        print(f"  post_mod_output: {lrm_module.post_mod_output is not None}")
        print(f"  total_mod: {lrm_module.total_mod is not None}")
        if lrm_module.total_mod is not None:
            print(f"    total_mod shape: {lrm_module.total_mod.shape}")
            print(f"    total_mod mean: {lrm_module.total_mod.mean():.4f}")

if __name__ == "__main__":
    test_forward_hooks() 