#!/usr/bin/env python3
"""
Fix LRM implementation by updating the forward_hook_target method to match the sample code.
"""

import os
import sys
import torch
import torch.nn.functional as F

def create_fixed_forward_hook_target():
    """Create the corrected forward_hook_target method based on sample code."""
    
    def forward_hook_target(self, module, input, output):
        '''modulate target output - FIXED VERSION'''
        
        if self.disable_modulation_during_inference:
            return output
        
        if len(self.mod_inputs) == 0:
            return output
        
        # we need to know current output size to adaptively resize in ModBlock
        target_size = output.shape[-2:] if len(output.shape) == 4 else 1
        
        # Get target layer name from the hook registration
        target_name = None
        for name, mod in self.model.named_modules():
            if mod is module:
                target_name = name
                break
        
        if target_name is None:
            return output
        
        # calculate long-range modulation to apply to output (sum across sources)
        total_mod = torch.zeros_like(output)
        found_active = False
        
        for mod_name, mod_module in self.named_children():
            # Filter only modules with names matching the "from_source_to_target" pattern
            if not mod_name.startswith("from_") or "_to_" not in mod_name:
                continue  # Skip unrelated modules
            
            # Extract source and target layer names - FIXED PARSING
            parts = mod_name.split('_')
            if len(parts) < 4:
                continue
                
            # Reconstruct source layer name (everything between 'from_' and '_to_')
            source_parts = []
            target_parts = []
            to_index = -1
            
            for i, part in enumerate(parts):
                if part == 'to':
                    to_index = i
                    break
                elif i > 0:  # Skip 'from'
                    source_parts.append(part)
            
            if to_index > 0:
                target_parts = parts[to_index + 1:]
            
            source_layer = '.'.join(source_parts)
            target_layer = '.'.join(target_parts)
            
            # Check if this modulation targets the current layer
            if target_layer == target_name.replace('.', '_'):
                if mod_name in self.mod_inputs:  # Ensure input exists for this source
                    source_activation = self.mod_inputs[mod_name]
                    
                    try:
                        # Apply ModBlock transformation
                        mod = mod_module(source_activation, target_size=target_size)
                        total_mod = total_mod + mod
                        found_active = True
                        
                        print(f"✅ Applied modulation: {source_layer} -> {target_layer}")
                        
                    except Exception as e:
                        print(f"❌ Modulation error in {mod_name}: {e}")
                        continue
        
        if not found_active:
            return output  # Return original output
        
        # Apply modulation (e.g., x = x + x * f) - SAME AS SAMPLE
        output = output + output * total_mod
        output = F.relu(output, inplace=False)
        
        return output
    
    return forward_hook_target

def patch_lrm_model(model):
    """Patch the LRM model with the fixed forward_hook_target method."""
    
    # Replace the forward_hook_target method
    import types
    fixed_method = create_fixed_forward_hook_target()
    model.lrm.forward_hook_target = types.MethodType(fixed_method, model.lrm)
    
    # Re-register hooks with the fixed method
    # First remove existing hooks
    for hook in model.lrm.targ_hooks:
        if hasattr(hook, 'remove'):
            hook.remove()
    model.lrm.targ_hooks.clear()
    
    # Re-register hooks with fixed method
    model_layers = dict([*model.named_modules()])
    for conn in model.lrm.mod_connections:
        target_name = conn['target']
        if target_name in model_layers:
            target_module = model_layers[target_name]
            hook = target_module.register_forward_hook(model.lrm.forward_hook_target)
            model.lrm.targ_hooks.append(hook)
    
    print("✅ LRM model patched with fixed forward_hook_target method")
    return model

if __name__ == "__main__":
    print("LRM Implementation Fix")
    print("This script provides the corrected forward_hook_target method.")
    print("Use patch_lrm_model(model) to apply the fix to your model.") 