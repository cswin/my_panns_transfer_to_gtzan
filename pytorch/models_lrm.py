"""
Long Range Modulation (LRM) implementation for emotion regression.

This module implements the original LRM feedback system for PANNs emotion regression,
based on the sophisticated implementation from the research codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
from torch.nn.modules.utils import _pair
from functools import partial
from typing import Any, Callable, List, Optional, Type, Union

from models import FeatureEmotionRegression_Cnn6, init_layer
from pytorch_utils import do_mixup


# ===================================================================
#  Original LRM Components
# ===================================================================

def init_linear(m, act_func=None, init='auto', bias_std=0.01):
    if getattr(m, 'bias', None) is not None and bias_std is not None: 
        torch.nn.init.normal_(m.bias, 0, bias_std)
    if init == 'auto':
        if act_func in (F.relu_, F.leaky_relu_):
            init = torch.nn.init.kaiming_uniform_
        else:
            init = getattr(act_func.__class__, '__default_init__', None)
        if init is None: 
            init = getattr(act_func, '__default_init__', None)
    if init is not None: 
        init(m.weight)


class MemoryFormatModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory_format = torch.contiguous_format

    def set_memory_format(self, memory_format):
        self.memory_format = memory_format

    def _apply(self, fn):
        for module in self.children():
            if isinstance(module, MemoryFormatModule):
                module.memory_format = self.memory_format
        return super()._apply(fn)


class ChannelNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        if len(x.shape) == 4:
            x = x.permute(0, 2, 3, 1)
            x = super().forward(x)
            x = x.permute(0, 3, 1, 2)
        elif len(x.shape) == 3:
            x = x.permute(0, 2, 1)
            x = super().forward(x)
            x = x.permute(0, 2, 1)
        else:
            x = super().forward(x)
        return x


class AdaptiveFullstackNorm(nn.Module):
    def __init__(self, normalized_shape=(), eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        # Adaptive normalization that works with any shape
        original_shape = x.shape
        x = x.view(x.shape[0], -1)  # Flatten all dims except batch
        
        # Compute mean and std across all dims except batch
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        x = x.view(original_shape)
        return x


class FeedbackScale(nn.Module):
    def __init__(self, mode='tanh'):
        super().__init__()
        self.mode = mode

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        if self.mode == 'tanh':
            x = torch.tanh(x)
        return x


class AdaptiveUpsample(nn.Module):
    def __init__(self, upsample_mode='AdaptiveAvgPool2d'):
        super().__init__()
        self.upsample_mode = upsample_mode

    def forward(self, x, out_size):
        if self.upsample_mode == "UpsampleBilinear":
            x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=True)
        elif self.upsample_mode == "UpsampleBicubic":
            x = F.interpolate(x, size=out_size, mode='bicubic', align_corners=True)
        elif self.upsample_mode == "AdaptiveAvgPool2d":
            x = F.adaptive_avg_pool2d(x, out_size)
        return x


class UpsampleBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, in_shape, out_shape, upsample_mode='UpsampleBilinear'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.upsample_mode = upsample_mode

        layers = OrderedDict([
            ('upsample', AdaptiveUpsample(upsample_mode=upsample_mode)),
            ('conv1x1', nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1, 1), bias=False))
        ])
        super().__init__(layers)

    def forward(self, input, out_size=None):
        for module in self:
            if hasattr(module, 'forward') and len(module.forward.__code__.co_varnames) >= 3:
                input = module(input, out_size)
            else:
                input = module(input)
        return input


class NormSquashResize(nn.Module):
    def __init__(self, in_channels, in_shape, out_shape, norm_type='ChannelNorm', scale_type='tanh',
                 resize_type='UpsampleBilinear'):
        super().__init__()
        
        if norm_type == 'ChannelNorm':
            self.norm = ChannelNorm(normalized_shape=(in_channels,))
        elif norm_type == 'AdaptiveFullstackNorm':
            # For AdaptiveFullstackNorm, use a simpler approach that handles dynamic shapes
            # We'll normalize over all dimensions except batch
            self.norm = AdaptiveFullstackNorm(normalized_shape=())  # Empty for adaptive
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")
            
        self.scale = FeedbackScale(mode=scale_type)
        self.resize = UpsampleBlock(in_channels, in_channels, in_shape, out_shape, resize_type)

    def forward(self, x, out_size):
        x = self.norm(x)
        x = self.scale(x)
        x = self.resize(x, out_size)
        return x


class ModBlock(MemoryFormatModule):
    """Original LRM ModBlock with sophisticated processing pipeline."""
    
    def __init__(self, name, in_channels, out_channels, in_shape, out_shape, 
                 block_order='norm-squash-resize', norm_type='ChannelNorm', 
                 resize_type='UpsampleBilinear'):
        super().__init__()
        
        self.name = name
        
        # Original LRM rescaling pipeline
        self.rescale = NormSquashResize(in_channels, in_shape, out_shape, scale_type='tanh',
                                      norm_type=norm_type, resize_type=resize_type)
        
        # Learnable asymmetric scaling parameters (same as original)
        self.neg_scale = torch.nn.Parameter(torch.FloatTensor([1.0]))
        self.pos_scale = torch.nn.Parameter(torch.FloatTensor([1.0]))
        
        # 1x1 convolution for channel mapping
        self.modulation = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=True)

    def forward(self, x, target_size):
        """
        Original LRM forward pass:
        1. Normalize, squash (tanh), and spatially interpolate
        2. Apply asymmetric scaling to negative and positive values
        3. Apply 1x1 convolution for channel mapping
        """
        # DEVICE SAFETY: Ensure input is on the same device as the module
        module_device = next(self.parameters()).device
        if x.device != module_device:
            x = x.to(module_device)
        
        # Normalize, squash, and spatially interpolate the feedback activations
        x = self.rescale(x, target_size)
        
        # Scale neg and pos separately to allow asymmetric inhibition/facilitation
        # (parameters are already on the correct device)
        neg_mask, pos_mask = x < 0, x >= 0
        x = x * (neg_mask.float() * self.neg_scale + pos_mask.float() * self.pos_scale)
        x = x.to(memory_format=self.memory_format)
        
        # 1x1 conv to map from source to target channels
        x = self.modulation(x)
        
        if target_size == 1:
            # Squeeze out the spatial dimensions
            x = x.squeeze(-1).squeeze(-1)
            
        return x


def get_layer_shapes(model, layer_names, input_tensor):
    """Extract output shapes for specified layers."""
    shapes = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, input, output):
            shapes[name] = output.shape
        return hook
    
    # Register hooks
    model_layers = dict(model.named_modules())
    for name in layer_names:
        if name in model_layers:
            hook = model_layers[name].register_forward_hook(make_hook(name))
            hooks.append(hook)
    
    # Forward pass to get shapes
    model.eval()
    with torch.no_grad():
        # For emotion models, we need mel-spectrogram input, not image input
        # The input should be (batch, time_steps, mel_bins)
        if input_tensor.dim() == 4 and input_tensor.shape[1] == 3:
            # Convert from image format to mel-spectrogram format
            batch_size = input_tensor.shape[0]
            time_steps = input_tensor.shape[2]  # Use height as time_steps
            mel_bins = 64  # Standard mel bins for emotion models
            input_tensor = torch.randn(batch_size, time_steps, mel_bins)
        
        _ = model(input_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
        
    return shapes


class LongRangeModulation(nn.Sequential):
    """Original LRM implementation with sophisticated steering capabilities."""
    
    def __init__(self, model, mod_connections, img_size=224):
        # Initialize parent class later
        self.targ_hooks = []
        self.mod_inputs = {}
        self.mod_hooks = []
        self.disable_modulation_during_inference = False
        
        # Get connection info
        if isinstance(mod_connections, list):
            # Convert list format to grouped format
            connections = defaultdict(list)
            for conn in mod_connections:
                connections[conn['target']].append(conn['source'])
            mod_connections = connections
        
        # We expect only one target for emotion regression
        if len(mod_connections) != 1:
            raise ValueError("Emotion LRM expects exactly one modulation target")
            
        mod_target = list(mod_connections.keys())[0]
        mod_sources = list(mod_connections.values())[0]
        
        self.name = f"{mod_target.replace('.', '_')}_modulation"
        
        # Get layer shapes using proper mel-spectrogram input
        layer_names = [mod_target] + mod_sources
        # For emotion models, use mel-spectrogram format: (batch, time_steps, mel_bins)
        mel_bins = 64  # Standard for emotion models
        time_steps = 1024  # Reasonable default
        x = torch.rand(1, time_steps, mel_bins)
        shapes = get_layer_shapes(model, layer_names, x)
        
        # Get model layers
        model_layers = dict(model.named_modules())
        
        # Register target hook
        target_module = model_layers[mod_target]
        self.targ_hooks.append(target_module.register_forward_hook(self.forward_hook_target))
        
        # Create ModBlocks for each source
        layers = OrderedDict()
        for source_layer_name in mod_sources:
            source_module = model_layers[source_layer_name]
            
            name = f'from_{source_layer_name.replace(".", "_")}_to_{mod_target.replace(".", "_")}'
            self.mod_hooks.append(source_module.register_forward_hook(partial(self.hook_fn, name=name)))
            
            source_size = _pair(shapes[source_layer_name][2:]) or (1, 1)
            target_size = _pair(shapes[mod_target][2:]) or (1, 1)
            
            modblock = ModBlock(
                name=name,
                in_channels=shapes[source_layer_name][1],
                out_channels=shapes[mod_target][1],
                in_shape=source_size,
                out_shape=target_size,
                block_order='norm-squash-resize',
                norm_type='AdaptiveFullstackNorm',
                resize_type='UpsampleBilinear'
            )
            layers[name] = modblock
        
        # Initialize parent class
        super().__init__(layers)
        
        # Clear initial mod_inputs
        self.mod_inputs = {}

    def forward_hook_target(self, module, input, output):
        """Apply modulation to target output."""
        
        if self.disable_modulation_during_inference or len(self.mod_inputs) == 0:
            return output
        
        # Get target size for adaptive resizing
        target_size = output.shape[-2:] if len(output.shape) == 4 else 1
        
        # Apply modulation from all sources
        total_mod = torch.zeros_like(output)
        
        for mod_name, mod_module in self.named_children():
            if mod_name in self.mod_inputs:
                source_activation = self.mod_inputs[mod_name]
                
                try:
                    # IMPROVED DEVICE FIX: Keep ModBlock on its original device
                    # Move tensors to ModBlock device, compute modulation, then move result back
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
                    
                except Exception as e:
                    # Enhanced error reporting for debugging
                    print(f"❌ Modulation error in {mod_name}:")
                    print(f"   Target output shape: {output.shape} (device: {output.device})")
                    print(f"   Source activation shape: {source_activation.shape} (device: {source_activation.device})")
                    print(f"   ModBlock device: {next(mod_module.parameters()).device}")
                    print(f"   Error: {e}")
                    # Continue without this modulation
                    continue
        
        # Apply modulation: output = output + output * modulation
        output = output + output * total_mod
        output = F.relu(output, inplace=False)
        
        return output

    def hook_fn(self, module, input, output, name):
        """Store source activation for modulation."""
        self.mod_inputs[name] = output

    def clear_stored_activations(self):
        """Clear stored activations."""
        self.mod_inputs.clear()

    def enable(self):
        """Enable modulation."""
        self.disable_modulation_during_inference = False

    def disable(self):
        """Disable modulation."""
        self.disable_modulation_during_inference = True

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.targ_hooks + self.mod_hooks:
            hook.remove()
        self.targ_hooks.clear()
        self.mod_hooks.clear()

    def adjust_modulation_strength(self, strength):
        """Adjust modulation strength for all ModBlocks."""
        if isinstance(strength, (int, float)):
            neg_strength = pos_strength = strength
        else:
            neg_strength, pos_strength = strength
        
        with torch.no_grad():
            for mod_block in self.children():
                # Store original values if not already stored
                if not hasattr(mod_block, 'neg_scale_orig'):
                    mod_block.neg_scale_orig = mod_block.neg_scale.clone()
                if not hasattr(mod_block, 'pos_scale_orig'):
                    mod_block.pos_scale_orig = mod_block.pos_scale.clone()
                
                # Apply strength adjustment
                mod_block.neg_scale.data = mod_block.neg_scale_orig * neg_strength
                mod_block.pos_scale.data = mod_block.pos_scale_orig * pos_strength

    def reset_modulation_strength(self):
        """Reset modulation strength to original values."""
        with torch.no_grad():
            for mod_block in self.children():
                if hasattr(mod_block, 'neg_scale_orig'):
                    mod_block.neg_scale.data = mod_block.neg_scale_orig.clone()
                    delattr(mod_block, 'neg_scale_orig')
                if hasattr(mod_block, 'pos_scale_orig'):
                    mod_block.pos_scale.data = mod_block.pos_scale_orig.clone()
                    delattr(mod_block, 'pos_scale_orig')

    def forward(self, x):
        """Forward pass placeholder - not used in hook-based implementation."""
        pass


class FeatureEmotionRegression_Cnn6_LRM(nn.Module):
    """
    Feature-based emotion regression with original LRM feedback implementation.
    
    This model uses the sophisticated original LRM architecture to enable top-down 
    feedback where emotion predictions from higher layers modulate intermediate 
    convolutional features through multiple forward passes.
    """
    
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, freeze_base=True, forward_passes=2):
        """
        Initialize LRM emotion regression model with original LRM implementation.
        
        Args:
            sample_rate: int, sample rate of audio
            window_size: int, window size for STFT
            hop_size: int, hop size for STFT  
            mel_bins: int, number of mel bins
            fmin: int, minimum frequency
            fmax: int, maximum frequency
            freeze_base: bool, whether to freeze the pretrained base model
            forward_passes: int, number of forward passes for iterative refinement
        """
        super(FeatureEmotionRegression_Cnn6_LRM, self).__init__()
        
        # Base emotion regression model
        self.base_model = FeatureEmotionRegression_Cnn6(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, freeze_base
        )
        
        # LRM configuration
        self.forward_passes = forward_passes
        
        # Original LRM approach: Direct feedback from final predictions to conv layers
        # Define modulation connections based on psychological principles:
        # - Valence (positive/negative) affects semantic interpretation -> higher-level features
        # - Arousal (energy/activation) affects attention to acoustic details -> lower-level features
        
        # For now, use a simple single-target approach for compatibility
        # Future enhancement: Multiple targets for valence/arousal pathway separation
        mod_connections = {
            'base.conv_block3': ['base.conv_block4']  # Conv4 modulates Conv3
        }
        
        # Initialize original LRM system
        self.lrm = LongRangeModulation(self.base_model, mod_connections, img_size=224)
    
    def forward(self, input, mixup_lambda=None, forward_passes=None, 
                return_all_passes=False, modulation_strength=None):
        """
        Forward pass with original LRM feedback implementation.
        
        Args:
            input: (batch_size, time_steps, mel_bins) pre-computed features
            mixup_lambda: float, mixup parameter
            forward_passes: int, number of forward passes
            return_all_passes: bool, return outputs from all passes
            modulation_strength: float or tuple, strength of modulation
                               If float: same strength for pos/neg
                               If tuple: (neg_strength, pos_strength)
            
        Returns:
            output_dict or list of output_dicts
        """
        # Handle DataParallel dimension issue and extra dimensions
        # DataParallel can add an extra dimension, so we need to handle multiple cases
        if input.dim() == 5:
            # DataParallel case: (num_gpus, batch_size, time_steps, mel_bins)
            # Reshape to (batch_size * num_gpus, time_steps, mel_bins)
            original_shape = input.shape
            input = input.view(-1, original_shape[-2], original_shape[-1])
        elif input.dim() == 4 and input.shape[1] == 1:
            # Extra dimension case: (batch_size, 1, time_steps, mel_bins)
            # Squeeze out the extra dimension
            input = input.squeeze(1)  # (batch_size, time_steps, mel_bins)
        elif input.dim() != 3:
            raise ValueError(f"Expected 3D, 4D (with dim 1 = 1), or 5D input, got {input.dim()}D input with shape {input.shape}")
        
        # Handle mixup
        if self.training and mixup_lambda is not None:
            input = do_mixup(input, mixup_lambda)
        
        # Determine number of passes
        num_passes = forward_passes if forward_passes is not None else self.forward_passes
        
        # Apply modulation strength if specified
        if modulation_strength is not None:
            self.lrm.adjust_modulation_strength(modulation_strength)
        
        all_outputs = []
        
        # Clear any previous stored activations before starting
        self.lrm.clear_stored_activations()
        
        # Original LRM approach: Multiple forward passes with automatic hook-based modulation
        for pass_idx in range(num_passes):
            # Forward pass through base model - LRM hooks will automatically apply modulation
            output = self.base_model(input, mixup_lambda=None)
            all_outputs.append(output)
            
            # The LRM hooks automatically store activations for the next pass
            # No manual intervention needed - this is the beauty of the original LRM design
        
        # Reset modulation strength if it was adjusted
        if modulation_strength is not None:
            self.lrm.reset_modulation_strength()
        
        # Return results
        if return_all_passes:
            return all_outputs
        else:
            # For segment-based processing, we want the final pass prediction
            # But the evaluation system will handle aggregation across segments
            return all_outputs[-1]
    
    def clear_feedback_state(self):
        """Clear feedback state between different audio files."""
        self.lrm.clear_stored_activations()
    
    def enable_feedback(self):
        """Enable LRM feedback."""
        self.lrm.enable()
    
    def disable_feedback(self):
        """Disable LRM feedback."""
        self.lrm.disable()
    
    def set_forward_passes(self, num_passes):
        """Set number of forward passes."""
        self.forward_passes = num_passes
    
    def set_modulation_strength(self, strength):
        """Set modulation strength."""
        self.lrm.adjust_modulation_strength(strength)
    
    def reset_modulation_strength(self):
        """Reset modulation strength to original values."""
        self.lrm.reset_modulation_strength()
    
    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights."""
        self.base_model.load_from_pretrain(pretrained_checkpoint_path)
    
    def __del__(self):
        """Cleanup hooks."""
        if hasattr(self, 'lrm'):
            self.lrm.remove_hooks() 