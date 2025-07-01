#!/usr/bin/env python3
"""
Fix LRM source hook system to properly handle external steering signals.
"""

import os
import sys
import torch
import numpy as np
import json
import h5py
from functools import partial

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def create_fixed_add_steering_signal():
    """Create a fixed add_steering_signal method that properly integrates with the hook system."""
    
    def add_steering_signal(self, source, activation, strength, alpha=1):
        """
        FIXED: Add steering signal that integrates properly with the LRM hook system.
        
        The key insight: Instead of manually populating mod_inputs, we need to create
        a temporary "virtual source layer" that the hooks can capture from.
        """
        
        # Map source to LRM layer names
        lrm_layer_names = self._map_source_layer_to_lrm(source)
        
        if not lrm_layer_names:
            print(f"⚠️ No LRM layers found for source: {source}")
            return
        
        # CRITICAL FIX: Store the steering signal in a way that the hook system can access
        # We create a "virtual activation" that gets injected during the forward pass
        
        # Store steering signals with metadata for proper application
        if not hasattr(self.lrm, 'steering_signals'):
            self.lrm.steering_signals = {}
        
        # Convert activation to proper format
        if isinstance(activation, (list, np.ndarray)):
            activation = torch.tensor(activation, dtype=torch.float32)
        
        # Ensure activation is on the correct device
        activation = activation.to(next(self.parameters()).device)
        
        # Format as 4D tensor for conv layers: (batch, channels, height, width)
        if activation.dim() == 1:
            activation = activation.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [128] -> [1, 128, 1, 1]
        elif activation.dim() == 2:
            activation = activation.unsqueeze(-1).unsqueeze(-1)  # [1, 128] -> [1, 128, 1, 1]
        
        # Store with strength and alpha for each target layer
        for lrm_layer_name in lrm_layer_names:
            self.lrm.steering_signals[lrm_layer_name] = {
                'activation': activation.clone(),
                'strength': strength,
                'alpha': alpha,
                'source': source
            }
        
        print(f"✅ Stored steering signal for {source} -> {len(lrm_layer_names)} target layers")
        
        # Apply strength adjustment to the relevant ModBlocks
        self._apply_strength_to_modblocks(lrm_layer_names, strength)
    
    return add_steering_signal

def create_fixed_forward_hook_target():
    """Create a fixed forward_hook_target that uses steering signals properly."""
    
    def forward_hook_target(self, module, input, output, target_name):
        """FIXED: Apply modulation using steering signals stored by add_steering_signal."""
        
        if self.disable_modulation_during_inference:
            return output
        
        # Check if we have steering signals for this target
        if not hasattr(self, 'steering_signals') or len(self.steering_signals) == 0:
            return output
        
        # Get target size for adaptive resizing
        target_size = output.shape[-2:] if len(output.shape) == 4 else 1
        
        # Apply modulation from steering signals that target this specific layer
        total_mod = torch.zeros_like(output)
        applied_count = 0
        
        for mod_name, mod_module in self.named_children():
            # Check if this ModBlock targets the current layer
            target_key = target_name.replace('.', '_')
            if target_key in mod_name and mod_name in self.steering_signals:
                steering_data = self.steering_signals[mod_name]
                source_activation = steering_data['activation']
                strength = steering_data['strength']
                alpha = steering_data['alpha']
                
                try:
                    # Apply ModBlock transformation with steering signal
                    mod_device = next(mod_module.parameters()).device
                    target_device = output.device
                    
                    # Move source activation to ModBlock device
                    source_on_mod_device = source_activation.to(mod_device)
                    
                    # Compute modulation on ModBlock device
                    mod = mod_module(source_on_mod_device, target_size=target_size)
                    
                    # Move modulation result to target device
                    if mod.device != target_device:
                        mod = mod.to(target_device)
                    
                    total_mod = total_mod + mod
                    applied_count += 1
                    
                    print(f"✅ Applied steering modulation: {mod_name} (strength={strength})")
                    
                except Exception as e:
                    print(f"❌ Modulation error in {mod_name}: {e}")
                    continue
        
        # Apply modulation: output = output + output * modulation
        if applied_count > 0:
            output = output + output * total_mod
            output = torch.relu(output, inplace=False)
            print(f"🎯 Total modulations applied to {target_name}: {applied_count}")
        
        return output
    
    return forward_hook_target

def create_fixed_clear_feedback_state():
    """Create a fixed clear_feedback_state that clears steering signals."""
    
    def clear_feedback_state(self):
        """FIXED: Clear feedback state including steering signals."""
        # Clear LRM stored activations
        if hasattr(self, 'lrm'):
            self.lrm.mod_inputs.clear()
            if hasattr(self.lrm, 'steering_signals'):
                self.lrm.steering_signals.clear()
        
        # Clear internal feedback storage
        self.valence_128d = None
        self.arousal_128d = None
    
    return clear_feedback_state

def apply_lrm_fixes(model):
    """Apply all LRM fixes to the model."""
    
    # Replace methods with fixed versions
    import types
    
    # Fix add_steering_signal
    fixed_add_steering_signal = create_fixed_add_steering_signal()
    model.add_steering_signal = types.MethodType(fixed_add_steering_signal, model)
    
    # Fix forward_hook_target
    fixed_forward_hook_target = create_fixed_forward_hook_target()
    model.lrm.forward_hook_target = types.MethodType(fixed_forward_hook_target, model.lrm)
    
    # Fix clear_feedback_state
    fixed_clear_feedback_state = create_fixed_clear_feedback_state()
    model.clear_feedback_state = types.MethodType(fixed_clear_feedback_state, model)
    
    # Add helper method for strength application
    def _apply_strength_to_modblocks(self, lrm_layer_names, strength):
        """Apply strength adjustment to specific ModBlocks."""
        for mod_name, mod_module in self.lrm.named_children():
            if any(layer_name in mod_name for layer_name in lrm_layer_names):
                # Store original values if not already stored
                if not hasattr(mod_module, 'neg_scale_orig'):
                    mod_module.neg_scale_orig = mod_module.neg_scale.clone()
                if not hasattr(mod_module, 'pos_scale_orig'):
                    mod_module.pos_scale_orig = mod_module.pos_scale.clone()
                
                # Apply strength adjustment
                with torch.no_grad():
                    mod_module.neg_scale.data = mod_module.neg_scale_orig * strength
                    mod_module.pos_scale.data = mod_module.pos_scale_orig * strength
    
    model._apply_strength_to_modblocks = types.MethodType(_apply_strength_to_modblocks, model)
    
    print("✅ Applied all LRM fixes to the model")
    return model

def test_fixed_lrm():
    """Test the fixed LRM implementation."""
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
    
    print("Testing with current implementation...")

if __name__ == "__main__":
    test_fixed_lrm() 