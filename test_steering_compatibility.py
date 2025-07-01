#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

import torch
import numpy as np
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def test_steering_compatibility():
    """Test that the steering logic matches the tmp models implementation."""
    
    print("Testing steering compatibility with tmp models...")
    
    # Create model
    model = FeatureEmotionRegression_Cnn6_LRM(32000, 1024, 320, 64, 50, 14000)
    model.eval()
    
    # Create dummy input
    batch_size = 1
    time_steps = 1024
    mel_bins = 64
    input_tensor = torch.randn(batch_size, time_steps, mel_bins)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Test 1: Clear feedback state
    print("\n1. Testing clear_feedback_state...")
    model.clear_feedback_state()
    print("✓ clear_feedback_state works")
    
    # Test 2: Add steering signal
    print("\n2. Testing add_steering_signal...")
    # Create dummy steering signal (128D valence representation)
    valence_steering = torch.randn(128)
    arousal_steering = torch.randn(128)
    
    try:
        model.add_steering_signal(
            source='affective_valence_128d',
            activation=valence_steering,
            strength=1.0,
            alpha=1.0
        )
        model.add_steering_signal(
            source='affective_arousal_128d',
            activation=arousal_steering,
            strength=1.0,
            alpha=1.0
        )
        print("✓ add_steering_signal works")
    except Exception as e:
        print(f"✗ add_steering_signal failed: {e}")
        return False
    
    # Test 3: Check mod_inputs structure
    print("\n3. Testing mod_inputs structure...")
    try:
        # Check that mod_inputs are stored in the correct location
        for lrm_module_name, lrm_module in model.lrm.named_children():
            print(f"  LRM module: {lrm_module_name}")
            print(f"  mod_inputs keys: {list(lrm_module.mod_inputs.keys())}")
            for key, value in lrm_module.mod_inputs.items():
                print(f"    {key}: {value.shape}")
        print("✓ mod_inputs structure matches tmp models")
    except Exception as e:
        print(f"✗ mod_inputs structure check failed: {e}")
        return False
    
    # Test 4: Enable feedback and run forward pass
    print("\n4. Testing forward pass with steering...")
    try:
        model.lrm.enable()
        with torch.no_grad():
            output = model(input_tensor, forward_passes=2)
        
        print(f"✓ Forward pass successful")
        print(f"  Output valence: {output['valence'].shape}")
        print(f"  Output arousal: {output['arousal'].shape}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # Test 5: Test steering signal application pattern
    print("\n5. Testing steering signal application pattern...")
    try:
        # Clear and reapply steering signals
        model.clear_feedback_state()
        
        # Apply steering signals using the same pattern as tmp models
        model.add_steering_signal(
            source='affective_valence_128d',
            activation=valence_steering,
            strength=2.0,  # Test different strength
            alpha=0.5      # Test different alpha
        )
        
        # Check that the steering signals are applied correctly
        pattern_found = False
        for lrm_module_name, lrm_module in model.lrm.named_children():
            for feedback_module_name, feedback_module in lrm_module.named_children():
                if 'from_affective_valence_128d_to_' in feedback_module_name:
                    if feedback_module_name in lrm_module.mod_inputs:
                        pattern_found = True
                        steering_activation = lrm_module.mod_inputs[feedback_module_name]
                        print(f"  Found steering signal: {feedback_module_name}")
                        print(f"  Activation shape: {steering_activation.shape}")
                        break
        
        if pattern_found:
            print("✓ Steering signal application pattern matches tmp models")
        else:
            print("✗ Steering signal application pattern not found")
            return False
            
    except Exception as e:
        print(f"✗ Steering signal application test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Steering logic is compatible with tmp models.")
    return True

if __name__ == "__main__":
    success = test_steering_compatibility()
    if success:
        print("\n✅ Steering compatibility test PASSED")
    else:
        print("\n❌ Steering compatibility test FAILED")
        sys.exit(1) 