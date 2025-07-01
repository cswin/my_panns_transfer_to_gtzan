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

from .cnn_models import FeatureEmotionRegression_Cnn6, init_layer
from src.utils.pytorch_utils import do_mixup


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
                 resize_type='UpsampleBilinear', dropout_rate=0.0):
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
        
        # Remove dropout - not needed for feedback connections
        self.dropout = None

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
    
    # For our specific case, we know the shapes of the conv blocks
    # This avoids the circular dependency issue during initialization
    if hasattr(model, 'visual_system'):
        # Hard-coded shapes for CNN6 conv blocks (these are standard)
        shapes.update({
            'visual_system.base.conv_block1': torch.Size([1, 64, 32, 16]),   # Typical after pooling
            'visual_system.base.conv_block2': torch.Size([1, 128, 16, 8]),  
            'visual_system.base.conv_block3': torch.Size([1, 256, 8, 4]),   
            'visual_system.base.conv_block4': torch.Size([1, 512, 4, 2]),   
        })
        return shapes
    
    # Fallback to dynamic shape extraction for other models
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
        
        try:
            _ = model(input_tensor)
        except AttributeError:
            # Handle case where model is not fully initialized yet
            # Use default shapes for CNN6
            shapes.update({
                'visual_system.base.conv_block1': torch.Size([1, 64, 32, 16]),
                'visual_system.base.conv_block2': torch.Size([1, 128, 16, 8]),  
                'visual_system.base.conv_block3': torch.Size([1, 256, 8, 4]),   
                'visual_system.base.conv_block4': torch.Size([1, 512, 4, 2]),   
            })
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
        
    return shapes


class MultiHeadAttentionFusion(nn.Module):
    def __init__(self, object_dim=1000, emotion_dim=8, attention_dim=512,
                 num_heads=4, visual_bias=0.0, emotion_bias=0.0, predict_vision_emotion=2):
        super(MultiHeadAttentionFusion, self).__init__()
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        self.visual_bias = visual_bias
        self.emotion_bias = emotion_bias

        assert attention_dim % num_heads == 0

        # Transform both object and emotion features to attention_dim
        self.object_transform = nn.Linear(object_dim, attention_dim)
        self.emotion_transform = nn.Linear(emotion_dim, attention_dim)

        self.query = nn.Linear(attention_dim, attention_dim)
        self.key = nn.Linear(attention_dim, attention_dim)
        self.value = nn.Linear(attention_dim, attention_dim)
        self.fc_out = nn.Linear(attention_dim, attention_dim)
        self.predict_vision_emotion = predict_vision_emotion

    def forward(self, object_feats, emotion_feats, predict_vision_emotion=None):
        """
        object_feats: Tensor of shape [batch_size, object_dim]
        emotion_feats: Tensor of shape [batch_size, emotion_dim]
        """
        predict_vision_emotion = self.predict_vision_emotion if predict_vision_emotion is None else predict_vision_emotion
        
        # Transform both object and emotion features to attention_dim
        transformed_object_feats = self.object_transform(object_feats)          # [batch_size, attention_dim]
        transformed_emotion_feats = self.emotion_transform(emotion_feats)       # [batch_size, attention_dim]

        # if not self.training:  # Only apply this logic during inference
        if predict_vision_emotion ==1:  # If only vision features are to be predicted
            return self.fc_out(transformed_object_feats)

        if predict_vision_emotion ==2: # If only emotion features are to be predicted
            return self.fc_out(transformed_emotion_feats)

        # Now both have shape [batch_size, attention_dim]
        combined_feats = torch.stack([transformed_object_feats, transformed_emotion_feats], dim=1)
        # combined_feats: [batch_size, 2, attention_dim]

        Q = self.query(combined_feats)
        K = self.key(combined_feats)
        V = self.value(combined_feats)

        Q = Q.view(Q.shape[0], 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(K.shape[0], 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(V.shape[0], 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attention_scores = torch.einsum("bhqd,bhkd->bhqk", Q, K) / (self.head_dim ** 0.5)
        # print("Visual bias:", self.visual_bias)
        # print("Emotion bias:", self.emotion_bias)

        # Shift columns differently to bias attention
        attention_scores[:, :, :, 0] += self.visual_bias  # vision key
        attention_scores[:, :, :, 1] += self.emotion_bias  # emotion key

        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
        attention_output = attention_output.permute(0, 2, 1, 3).reshape(attention_output.shape[0], 2, -1)
        fused = torch.mean(attention_output, dim=1)

        return self.fc_out(fused)


class LongRangeModulation(nn.Sequential):
    def __init__(self, model, mod_connections, img_size=224):
        # Initialize parent class later
        self.targ_hooks = []
        self.mod_inputs = {}
        self.mod_hooks = []
        self.remove_mod_inputs = False
        self.disable_modulation_during_inference = False
        
        # Store the original mod_connections for reference
        self.mod_connections = mod_connections
        
        # Group connections by destination
        connections = defaultdict(list)
        for conn in mod_connections:
            connections[conn['destination']].append(conn['source'])
        
        # Create modulation layers for each destination
        layers = OrderedDict([])
        for dest_layer, source_layers in connections.items():
            mod_layer = LongRangeModulationSingle(
                model, dest_layer, source_layers, img_size=img_size
            )
            layers[mod_layer.name] = mod_layer
        
        super().__init__(layers)
        
        # Clear mod_inputs created when initializing modules
        for block in self.children():
            block.mod_inputs = {}
    
    def forward_hook_target(self, module, input, output, target_name):
        """Forward hook for target layer - not used in this implementation."""
        pass

    def hook_fn(self, module, input, output, name):
        """Hook function to store activations."""
        self.mod_inputs[name] = output

    def clear_stored_activations(self):
        """Clear stored activations."""
        self.mod_inputs.clear()
        for block in self.children():
            block.mod_inputs.clear()

    def enable(self):
        """Enable modulation."""
        self.disable_modulation_during_inference = False
        for block in self.children():
            block.disable_modulation_during_inference = False

    def disable(self):
        """Disable modulation."""
        self.disable_modulation_during_inference = True
        for block in self.children():
            block.disable_modulation_during_inference = True

    def remove_hooks(self):
        """Remove all hooks."""
        for hook_item in self.targ_hooks:
            if isinstance(hook_item, tuple):
                hook_item[0].remove()  # Remove the actual hook object
            else:
                hook_item.remove()
        for hook in self.mod_hooks:
            hook.remove()
        self.targ_hooks.clear()
        self.mod_hooks.clear()
        
        # Remove hooks from children
        for block in self.children():
            block.remove_hooks()

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
        for lrm_module_name, lrm_module in self.named_children():
            for feedback_module_name, feedback_module in lrm_module.named_children():
                if hasattr(feedback_module, 'neg_scale_orig'):
                    feedback_module.neg_scale = nn.Parameter(feedback_module.neg_scale_orig)
                    del feedback_module.neg_scale_orig
                if hasattr(feedback_module, 'pos_scale_orig'):
                    feedback_module.pos_scale = nn.Parameter(feedback_module.pos_scale_orig)
                    del feedback_module.pos_scale_orig

    def forward(self, x):
        """Forward pass placeholder - not used in hook-based implementation."""
        pass


class LongRangeModulationSingle(nn.Sequential):
    """Single destination LongRangeModulation implementation matching tmp/models/layers_emotion.py"""
    
    def __init__(self, model, mod_target, mod_sources, img_size=224,
                 mod_block_order='norm-squash-resize',
                 mod_norm_type='AdaptiveFullstackNorm',
                 mod_resize_type='UpsampleBilinear',
                 active_connections=None, is_fusion=False,
                 attention_dim=512,
                 num_heads=4, visual_bias=0, emotion_bias=0.0,
                 predict_vision_emotion=2, inactive_source_layer=None):
        
        # Initialize parent class later
        self.targ_hooks = []
        self.mod_inputs = {}
        self.mod_hooks = []
        self.remove_mod_inputs = False
        self.predict_vision_emotion = predict_vision_emotion
        self.inactive_source_layer = inactive_source_layer
        self.disable_modulation_during_inference = False

        # If active_connections is None, consider all connections as active
        self.active_connections = active_connections if active_connections is not None else [
            (source, mod_target) for source in mod_sources
        ]

        self.name = f"{mod_target.replace('.', '_')}_modulation"

        # first get the output shapes for mod_target and mod_sources
        layer_names = [mod_target] + mod_sources
        # Use mel-spectrogram input format for emotion models
        mel_bins = 64  # Standard for emotion models
        time_steps = 1024  # Reasonable default
        x = torch.rand(1, time_steps, mel_bins)
        shapes = get_layer_shapes(model, layer_names, x)

        # get list of modules from model
        model_layers = dict([*model.named_modules()])

        # register a forward hook for the target_module
        # this is where we'll modulate the target_module's output
        target_module = model_layers[mod_target]

        # The selected line of code registers a forward hook on the target_module. This hook will call the forward_hook_target method whenever the target_module performs a forward pass.
        self.targ_hooks += [target_module.register_forward_hook(self.forward_hook_target)]

        # iterate over modulation sources, adding a ModBlock for each
        layers = OrderedDict([])
        for source_layer_name in mod_sources:
            # Handle custom source names that don't exist in the model
            if source_layer_name in model_layers:
                source_module = model_layers[source_layer_name]
            else:
                # For custom sources like 'affective_valence_128d', we'll create a placeholder
                # The actual activation will be set by steering signals
                source_module = None

            name = f'from_{source_layer_name.replace(".", "_")}_to_{mod_target.replace(".", "_")}'
            
            # Only register hooks for actual model layers
            if source_module is not None:
                # The code on line 430 registers a forward hook for the `source_module`. This hook will call the `hook_fn` method with the specified `name` whenever the `source_module` performs a forward pass. The `partial` function is used to pass the `name` argument to the `hook_fn` method.
                # In summary, this line of code is setting up a mechanism to capture the output of the `source_module` during its forward pass and store it in the `mod_inputs` dictionary with the given `name` as the key.
                self.mod_hooks += [source_module.register_forward_hook(partial(self.hook_fn, name=name))]

            # Get shapes for source and target
            if source_layer_name in shapes:
                source_size = _pair(shapes[source_layer_name][2:]) or (1, 1)
                source_channels = shapes[source_layer_name][1]
            else:
                # For custom sources, use default values
                source_size = (1, 1)
                source_channels = 128  # Default for 128D affective representations

            target_size = _pair(shapes[mod_target][2:]) or (1, 1)
            target_channels = shapes[mod_target][1]

            mod_block_params = dict(
                name=name,
                source_channels=source_channels,
                target_channels=target_channels,
                source_size=source_size,
                target_size=target_size,
                mod_block_order=mod_block_order,
                mod_norm_type=mod_norm_type,
                mod_resize_type=mod_resize_type
            )

            # ModBlock handles aligning the source activation to the target activation
            # Normalizing, Squashing, and Resizing, then conv1x1 source => target
            modblock = ModBlock(name=mod_block_params['name'],
                                in_channels=mod_block_params['source_channels'],
                                out_channels=mod_block_params['target_channels'],
                                in_shape=mod_block_params['source_size'],
                                out_shape=mod_block_params['target_size'],
                                block_order=mod_block_params['mod_block_order'],
                                norm_type=mod_block_params['mod_norm_type'],
                                resize_type=mod_block_params['mod_resize_type'])
            layers[name] = modblock

        # Identity blocks for storing outputs
        self.pre_mod_output = None
        self.total_mod = None
        self.post_mod_output = None
        super().__init__(layers)

        # Initialize Multi-Head Attention Fusion if is_fusion is True
        self.is_fusion = is_fusion
        if len(mod_sources) != 2 and is_fusion:
            self.is_fusion = False

        if self.is_fusion:
            # Create a dedicated ModBlock for fused features
            fused_block_name = "from_fused_to_" + mod_target.replace('.', '_')
            fused_block = ModBlock(
                name=fused_block_name,
                in_channels=attention_dim,
                out_channels=shapes[mod_target][1],
                in_shape=(1, 1),  # Since fused_features is [B, C, 1, 1] after unsqueeze
                out_shape=_pair(shapes[mod_target][2:]) or (1, 1),
                block_order=mod_block_order,
                norm_type=mod_norm_type,
                resize_type=mod_resize_type
            )
            self.add_module(fused_block_name, fused_block)
            self.fused_block_name = fused_block_name

            # We assume exactly two sources for fusion: first as "object", second as "emotion"
            self.multihead_fusion = MultiHeadAttentionFusion(
                object_dim=shapes[mod_sources[0]][1],
                emotion_dim=shapes[mod_sources[1]][1],
                attention_dim=attention_dim,
                num_heads=num_heads,
                visual_bias=visual_bias,
                emotion_bias=emotion_bias, predict_vision_emotion=predict_vision_emotion
            )
        else:
            self.multihead_fusion = None

    def update_active_connections(self, active_connections):
        """
        Update the active connections during inference.
        """
        self.active_connections = active_connections

    def forward_hook_target(self, module, input, output):
        '''modulate target output'''

        if self.disable_modulation_during_inference:
            self.pre_mod_output = output.clone()
            self.post_mod_output = output.clone()
            return output

        if len(self.mod_inputs) == 0:
            self.pre_mod_output = output.clone()
            self.post_mod_output = output.clone()
            return output

        # Save pre-modulation output
        self.pre_mod_output = output.clone()

        # we need to know current output size to adaptively resize in ModBlock
        target_size = output.shape[-2:] if len(output.shape) == 4 else 1

        # Initialize a list to collect activations from all source layers
        source_activations = []

        for mod_name, mod_module in self.named_children():
            # Filter only modules with names matching the "from_source_to_target" pattern
            if not mod_name.startswith("from_") or "_to_" not in mod_name:
                continue  # Skip unrelated modules

            # Extract source and target layer names
            # Handle module names like: from_affective_valence_128d_to_visual_system_base_conv_block4
            parts = mod_name.split('_to_')
            if len(parts) != 2:
                continue
                
            source_part = parts[0].replace('from_', '')  # Remove 'from_' prefix
            target_part = parts[1]
            
            # Convert underscores back to dots for layer names - but be selective
            # For source: handle affective sources specially
            if source_part.startswith('affective_'):
                # Keep affective sources as-is (don't convert underscores to dots)
                source_layer = source_part
            else:
                # For other sources, convert underscores to dots
                source_layer = source_part.replace('_', '.')
            
            # For target: visual_system_base_conv_block4 -> visual_system.base.conv_block4
            # Only convert specific underscores to dots based on known patterns
            if target_part.startswith('visual_system_base_'):
                # visual_system_base_conv_block4 -> visual_system.base.conv_block4
                target_layer = target_part.replace('visual_system_base_', 'visual_system.base.')
            else:
                # Fallback: convert all underscores to dots
                target_layer = target_part.replace('_', '.')

            # Check if the source-target pair is active
            if (source_layer, target_layer) in self.active_connections:

                if self.inactive_source_layer is not None and source_layer in self.inactive_source_layer:
                    continue

                if mod_name in self.mod_inputs:  # Ensure input exists for this source
                    source_activation = self.mod_inputs[mod_name]
                    # Append source activation to the list
                    source_activations.append(source_activation)

        # calculate long-range modulation to apply to output (sum across sources)
        total_mod = torch.zeros_like(output)
        # Compute modulation
        if self.is_fusion and self.multihead_fusion is not None and len(source_activations) == 2:
            # Use Multi-Head Attention to fuse the two source activations
            if len(source_activations) != 2:
                raise ValueError("Fusion mode requires exactly two active source activations.")

            # Global average pooling to reduce feature maps to [B, C]
            object_feats = (
                source_activations[0].mean(dim=[2, 3]) if source_activations[0].ndim == 4 else
                source_activations[0].mean(dim=[1, 2]) if source_activations[0].ndim == 3 else
                source_activations[0]
            )
            object_feats = object_feats.unsqueeze(0) if object_feats.ndim ==1 else object_feats

            emotion_feats = (
                source_activations[1].mean(dim=[2, 3]) if source_activations[1].ndim == 4 else
                source_activations[1].mean(dim=[1, 2]) if source_activations[1].ndim == 3 else
                source_activations[1]
            )

            # Apply Multi-Head Attention Fusion
            fused_features = self.multihead_fusion(object_feats, emotion_feats)  # [B, attention_dim]

            # Apply the dedicated fused ModBlock
            fused_block = self._modules[self.fused_block_name]
            mod_fused_features = fused_block(fused_features, target_size=target_size)

            total_mod = total_mod + mod_fused_features

        else:
            # Original non-fusion summation logic
            total_mod = torch.zeros_like(output)
            found_active = False
            for mod_name, mod_module in self.named_children():
                if not mod_name.startswith("from_") or "_to_" not in mod_name:
                    continue

                # Handle module names like: from_affective_valence_128d_to_visual_system_base_conv_block4
                parts = mod_name.split('_to_')
                if len(parts) != 2:
                    continue
                    
                source_part = parts[0].replace('from_', '')  # Remove 'from_' prefix
                target_part = parts[1]
                
                # Convert underscores back to dots for layer names - but be selective
                # For source: handle affective sources specially
                if source_part.startswith('affective_'):
                    # Keep affective sources as-is (don't convert underscores to dots)
                    source_layer = source_part
                else:
                    # For other sources, convert underscores to dots
                    source_layer = source_part.replace('_', '.')
                
                # For target: visual_system_base_conv_block4 -> visual_system.base.conv_block4
                # Only convert specific underscores to dots based on known patterns
                if target_part.startswith('visual_system_base_'):
                    # visual_system_base_conv_block4 -> visual_system.base.conv_block4
                    target_layer = target_part.replace('visual_system_base_', 'visual_system.base.')
                else:
                    # Fallback: convert all underscores to dots
                    target_layer = target_part.replace('_', '.')

                if (source_layer, target_layer) in self.active_connections and mod_name in self.mod_inputs:
                    if self.inactive_source_layer is not None and source_layer in self.inactive_source_layer:
                        continue
                    source_activation = self.mod_inputs[mod_name]
                    
                    mod = mod_module(source_activation, target_size=target_size)

                    total_mod = total_mod + mod
                    found_active = True

            if not found_active:
                self.pre_mod_output = output.clone()
                self.post_mod_output = output.clone()
                return output  # Return original output

        # Save the total modulation applied
        self.total_mod = total_mod.clone()
        
        # FIXED AMPLIFICATION: Apply consistent moderate scaling
        # This amplifies all steering effects while preventing saturation
        # MODULATION_AMPLIFIER = 3.0  # Moderate amplification for all strengths
        # total_mod = total_mod * MODULATION_AMPLIFIER
        
        # # Clamp to prevent extreme saturation but allow strong effects
        # total_mod = torch.clamp(total_mod, -1.0, 1.0)
        
        # Apply modulation (e.g., x = x + x * f)
        output = output + output * total_mod

        # Save post-modulation output
        self.post_mod_output = output.clone()
        output = F.relu(output, inplace=False)

        return output

    def hook_fn(self, module, input, output, name):
        self.mod_inputs[name] = output

    def remove_hooks(self):
        """Remove all hooks."""
        for hook_item in self.targ_hooks:
            if isinstance(hook_item, tuple):
                hook_item[0].remove()  # Remove the actual hook object
            else:
                hook_item.remove()
        for hook in self.mod_hooks:
            hook.remove()
        self.targ_hooks.clear()
        self.mod_hooks.clear()

    def forward(self, x):
        '''forward pass of lrm modules doesn't get called'''
        pass


class FeatureEmotionRegression_Cnn6_LRM(nn.Module):
    """
    Feature-based emotion regression with LRM feedback implementation.
    
    This model has the SAME structure as FeatureEmotionRegression_Cnn6_NewAffective but WITH feedback connections.
    It enables top-down feedback where emotion predictions from affective pathways modulate 
    intermediate convolutional features in the visual system through multiple forward passes.
    
    Architecture:
    - Visual System: Frozen CNN6 backbone (same as NewAffective)
    - Affective System: Separate valence/arousal pathways (same as NewAffective) 
    - Feedback: 128D affective representations modulate visual conv layers (LRM only)
    """
    
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, 
                 fmax, freeze_base=True, forward_passes=2):
        """
        Initialize LRM emotion regression model.
        
        Args:
            sample_rate: int, sample rate of audio
            window_size: int, window size for STFT
            hop_size: int, hop size for STFT  
            mel_bins: int, number of mel bins
            fmin: int, minimum frequency
            fmax: int, maximum frequency
            freeze_base: bool, whether to freeze the pretrained base model (always True for fair comparison)
            forward_passes: int, number of forward passes for iterative refinement
        """
        super(FeatureEmotionRegression_Cnn6_LRM, self).__init__()
        
        # Visual system - same frozen CNN6 backbone as NewAffective
        from models import FeatureEmotionRegression_Cnn6
        self.visual_system = FeatureEmotionRegression_Cnn6(
            sample_rate, window_size, hop_size, mel_bins, fmin, fmax, 
            freeze_base=True  # Always frozen for fair comparison with NewAffective
        )
        
        # Remove the regression heads from visual system since we'll use separate affective pathways
        # Keep only the visual processing (conv layers + fc1 → 512D embedding)
        del self.visual_system.fc_valence
        del self.visual_system.fc_arousal
        
        # New affective system - separate pathways for valence and arousal (same as NewAffective)
        self.affective_valence = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.affective_arousal = nn.Sequential(
            nn.Linear(512, 256), 
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Initialize the affective pathways
        from models import init_layer
        for pathway in [self.affective_valence, self.affective_arousal]:
            for layer in pathway:
                if isinstance(layer, nn.Linear):
                    init_layer(layer)
        
        # LRM configuration
        self.forward_passes = forward_passes
        
        # Feedback connections: 128D affective representations modulate visual conv layers
        # 4-way feedback connections: valence modulates higher layers, arousal modulates lower layers
        mod_connections = [
            # Valence modulates higher-level conv layers (semantic processing)
            {'source': 'affective_valence_128d', 'destination': 'visual_system.base.conv_block3'},
            {'source': 'affective_valence_128d', 'destination': 'visual_system.base.conv_block4'},
            # Arousal modulates lower-level conv layers (attention to acoustic details)
            {'source': 'affective_arousal_128d', 'destination': 'visual_system.base.conv_block1'},
            {'source': 'affective_arousal_128d', 'destination': 'visual_system.base.conv_block2'}
        ]
        
        # Storage for 128D feedback signals
        self.valence_128d = None
        self.arousal_128d = None
        
        # Initialize LRM system (after model is fully constructed)
        self.lrm = LongRangeModulation(self, mod_connections, img_size=224)
    
    def forward(self, input, mixup_lambda=None, forward_passes=None, 
                return_all_passes=False, modulation_strength=None, external_feedback=None,
                steering_signals=None, first_pass_steering=False, active_connections=None,
                return_list=False):
        """
        Forward pass with LRM feedback implementation for full-length audios.
        
        Args:
            input: (batch_size, time_steps, mel_bins) pre-computed features from full audio
            mixup_lambda: float, mixup parameter
            forward_passes: int, number of forward passes
            return_all_passes: bool, return outputs from all passes (legacy parameter)
            modulation_strength: float or tuple, strength of modulation
                               If float: same strength for pos/neg
                               If tuple: (neg_strength, pos_strength)
            external_feedback: dict, external feedback signals for steering (legacy parameter)
                             Format: {'valence': tensor, 'arousal': tensor}
                             Each tensor shape: (batch_size, 128)
            
            # Full steering interface (compatible with Test_Steering_Emotion.py)
            steering_signals: list of dicts, external steering signals
                            Format: [{'source': layer_name, 'activation': tensor, 
                                    'strength': float, 'alpha': float}]
            first_pass_steering: bool, whether to apply steering in first pass (priming)
            active_connections: list of tuples, specific connections to activate
                              Format: [('source_layer', 'target_layer'), ...]
            return_list: bool, whether to return list of outputs from all passes
            
        Returns:
            output_dict or list of output_dicts
        """
        # Handle DataParallel dimension issue and extra dimensions
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
            from utils.pytorch_utils import do_mixup
            input = do_mixup(input, mixup_lambda)
        
        # Determine number of passes
        num_passes = forward_passes if forward_passes is not None else self.forward_passes
        
        # Apply modulation strength if specified
        if modulation_strength is not None:
            self.lrm.adjust_modulation_strength(modulation_strength)
        
        # Handle active connections - filter LRM connections if specified
        if active_connections is not None:
            self._set_active_connections(active_connections)
        
        all_outputs = []
        
        # Apply external steering signals if provided
        # These persist across all passes and provide consistent external guidance
        if steering_signals is not None:
            self._apply_steering_signals(steering_signals)
        
        # DON'T clear stored activations here - we need them for multi-pass feedback
        # Each pass will generate new activations that are used by the next pass
        # Only clear activations between different audio samples (in clear_feedback_state)
        
        # For full-length audios, we compute feedback signals once and reuse them
        # This ensures consistent feedback across all passes since there are no segments
        feedback_computed = False
        stored_valence_128d = None
        stored_arousal_128d = None
        
        # Multiple forward passes with feedback
        for pass_idx in range(num_passes):
            # Apply external steering signals if provided
            apply_steering = False
            if steering_signals is not None:
                # Apply steering in first pass if first_pass_steering is True
                # Or apply steering in subsequent passes (default behavior)
                if first_pass_steering or pass_idx > 0:
                    apply_steering = True
                    # Steering signals already applied above, no need to apply again
            
            # Visual processing through frozen CNN6 backbone
            visual_embedding = self._forward_visual_system(input, mixup_lambda=None)
            
            # Affective processing through separate pathways
            # Extract 128D representations for feedback (second-to-last layer)
            valence_256d = self.affective_valence[0:2](visual_embedding)  # Linear(512,256) + ReLU
            valence_128d = self.affective_valence[2:4](valence_256d)      # Linear(256,128) + ReLU
            valence_out = self.affective_valence[4](valence_128d)         # Linear(128,1)
            
            arousal_256d = self.affective_arousal[0:2](visual_embedding)  # Linear(512,256) + ReLU
            arousal_128d = self.affective_arousal[2:4](arousal_256d)      # Linear(256,128) + ReLU
            arousal_out = self.affective_arousal[4](arousal_128d)         # Linear(128,1)
            
            # For full-length audios: compute feedback signals once and reuse
            if not feedback_computed:
                # Store 128D feedback signals for LRM modulation (computed once)
                self.valence_128d = valence_128d
                self.arousal_128d = arousal_128d
                stored_valence_128d = valence_128d
                stored_arousal_128d = arousal_128d
                feedback_computed = True
            else:
                # Use stored feedback signals for consistency across passes
                self.valence_128d = stored_valence_128d
                self.arousal_128d = stored_arousal_128d
            
            # Create output for this pass
            output = {
                'valence': valence_out,
                'arousal': arousal_out,
                'embedding': visual_embedding
            }
            all_outputs.append(output)
            
            # Store feedback signals for next pass (if not the last pass)
            if pass_idx < num_passes - 1:
                # Priority: steering_signals > external_feedback > internal predictions
                if steering_signals is not None and not first_pass_steering:
                    # External steering signals will be applied in next pass
                    # Still store internal feedback for LRM modulation
                    self._store_feedback_signals(stored_valence_128d, stored_arousal_128d)
                elif external_feedback is not None:
                    # Use legacy external feedback if provided
                    ext_valence_128d = external_feedback.get('valence', stored_valence_128d)
                    ext_arousal_128d = external_feedback.get('arousal', stored_arousal_128d)
                    self._store_feedback_signals(ext_valence_128d, ext_arousal_128d)
                else:
                    # Use stored internal predictions for consistency (full-length audio approach)
                    self._store_feedback_signals(stored_valence_128d, stored_arousal_128d)
        
        # Reset modulation strength if it was adjusted
        if modulation_strength is not None:
            self.lrm.reset_modulation_strength()
        
        # Reset active connections if they were modified
        if active_connections is not None:
            self._reset_active_connections()
        
        # Return results based on interface preference
        if return_list or return_all_passes:
            return all_outputs
        else:
            return all_outputs[-1]
    
    def _forward_visual_system(self, input, mixup_lambda=None):
        """Forward through visual system only (without regression heads)."""
        # Get the visual embedding (512D) from the frozen CNN6 backbone
        # We need to manually forward through the visual system but stop before fc_valence/fc_arousal
        
        # This is tricky because we deleted the fc_valence/fc_arousal, so we need to 
        # reconstruct the forward pass manually
        x = input.unsqueeze(1)  # Add channel dimension
        
        # Apply batch norm
        x = x.transpose(1, 3)
        x = self.visual_system.base.bn0(x)
        x = x.transpose(1, 3)
        
        if self.training:
            x = self.visual_system.base.spec_augmenter(x)

        # Mixup on spectrogram
        if self.training and mixup_lambda is not None:
            from utils.pytorch_utils import do_mixup
            x = do_mixup(x, mixup_lambda)

        # Forward through convolutional blocks
        x = self.visual_system.base.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.visual_system.base.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.visual_system.base.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.visual_system.base.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        
        # Global pooling
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.visual_system.base.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        
        return embedding
    
    def _store_feedback_signals(self, valence_128d, arousal_128d):
        """Store 128D feedback signals for LRM modulation."""
        # Store feedback signals for each ModBlock - need to match the ModBlock names
        # Reshape to add spatial dimensions for conv layer modulation
        valence_4d = valence_128d.unsqueeze(-1).unsqueeze(-1)  # (batch, 128, 1, 1)
        arousal_4d = arousal_128d.unsqueeze(-1).unsqueeze(-1)  # (batch, 128, 1, 1)
        
        # Store feedback signals for each target layer
        for conn in self.lrm.mod_connections:
            source_name = conn['source']
            target_name = conn['destination']
            mod_name = f'from_{source_name.replace(".", "_")}_to_{target_name.replace(".", "_")}'
            
            # Find the corresponding LRM module and store the signal
            for lrm_module_name, lrm_module in self.lrm.named_children():
                if mod_name in lrm_module.mod_inputs:
                    if 'affective_valence_128d' in source_name:
                        lrm_module.mod_inputs[mod_name] = valence_4d
                    elif 'affective_arousal_128d' in source_name:
                        lrm_module.mod_inputs[mod_name] = arousal_4d
    
    def clear_feedback_state(self):
        """Clear feedback state between different audio files."""
        # This should clear ALL stored activations (both internal feedback and steering)
        # because we're moving to a completely different audio sample
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
    
    def reset_modulation_strengths(self):
        """
        Reset modulation strengths to original values.
        """
        for lrm_module_name, lrm_module in self.lrm.named_children():
            for feedback_module_name, feedback_module in lrm_module.named_children():
                if hasattr(feedback_module, 'neg_scale_orig'):
                    feedback_module.neg_scale = nn.Parameter(feedback_module.neg_scale_orig)
                    del feedback_module.neg_scale_orig
                if hasattr(feedback_module, 'pos_scale_orig'):
                    feedback_module.pos_scale = nn.Parameter(feedback_module.pos_scale_orig)
                    del feedback_module.pos_scale_orig
    
    def load_from_pretrain(self, pretrained_checkpoint_path):
        """Load pretrained weights for visual system."""
        self.visual_system.load_from_pretrain(pretrained_checkpoint_path)
    
    def __del__(self):
        """Cleanup hooks."""
        if hasattr(self, 'lrm'):
            self.lrm.remove_hooks()

    def set_external_feedback(self, valence_128d=None, arousal_128d=None):
        """
        Set external feedback signals for steering the model behavior.
        
        This method allows researchers to inject custom feedback signals from external sources
        instead of using the model's own predictions for feedback.
        
        Args:
            valence_128d: torch.Tensor, shape (batch_size, 128)  
                         External valence representation for steering
            arousal_128d: torch.Tensor, shape (batch_size, 128)
                         External arousal representation for steering
        """
        if valence_128d is not None:
            self._store_external_feedback('valence', valence_128d)
        if arousal_128d is not None:
            self._store_external_feedback('arousal', arousal_128d)
    
    def _store_external_feedback(self, signal_type, signal_128d):
        """Store external feedback signal for specific emotion dimension."""
        # Reshape to add spatial dimensions for conv layer modulation
        signal_4d = signal_128d.unsqueeze(-1).unsqueeze(-1)  # (batch, 128, 1, 1)
        
        # Store feedback signals for each target layer
        for conn in self.lrm.mod_connections:
            source_name = conn['source']
            target_name = conn['destination']
            mod_name = f'from_{source_name.replace(".", "_")}_to_{target_name.replace(".", "_")}'
            
            if signal_type == 'valence' and 'affective_valence_128d' in source_name:
                self.lrm.mod_inputs[mod_name] = signal_4d
            elif signal_type == 'arousal' and 'affective_arousal_128d' in source_name:
                self.lrm.mod_inputs[mod_name] = signal_4d
    
    def _apply_steering_signals(self, steering_signals):
        """
        Apply external steering signals to specified layers.
        
        Args:
            steering_signals: list of dicts with format:
                            [{'source': layer_name, 'activation': tensor, 
                              'strength': float, 'alpha': float}]
        """
        # Reset modulation strengths to prevent accumulation
        self.reset_modulation_strengths()
        
        for signal_dict in steering_signals:
            source_layer = signal_dict['source']
            activation = signal_dict['activation']
            strength = signal_dict.get('strength', 1.0)
            alpha = signal_dict.get('alpha', 1.0)  # Blending ratio
            
            # Map source layer to our LRM layer names
            lrm_layer_name = self._map_source_layer_to_lrm(source_layer)
            if lrm_layer_name:
                # Apply steering with strength and alpha blending
                self._inject_steering_activation(lrm_layer_name, activation, strength, alpha)
    
    def _map_source_layer_to_lrm(self, source_layer):
        """
        Map external layer names to internal LRM layer structure.
        
        This is a mapping function that translates external layer names
        (like 'classifier.6') to our internal LRM feedback structure.
        """
        # For emotion regression, we need to map classifier layers to affective pathways
        if 'classifier' in source_layer or 'fc' in source_layer:
            # Map classifier/fc layers to both valence and arousal pathways
            return ['affective_valence_128d', 'affective_arousal_128d']
        elif 'conv' in source_layer or 'features' in source_layer:
            # Map conv/feature layers to visual system layers
            return [f'visual_system.base.{source_layer}']
        else:
            # Default mapping - try to use as-is
            return [source_layer]
    
    def _inject_steering_activation(self, lrm_layer_names, activation, strength, alpha):
        """
        Inject steering activation into LRM feedback system.
        
        Args:
            lrm_layer_names: list of LRM layer names to inject into
            activation: external activation tensor
            strength: steering strength multiplier
            alpha: blending ratio (1.0 = full external, 0.0 = full internal)
        """
        if not isinstance(lrm_layer_names, list):
            lrm_layer_names = [lrm_layer_names]
        
        # Handle strength parameter (can be float or tuple)
        if isinstance(strength, (int, float)):
            neg_scale_adjust, pos_scale_adjust = strength, strength
        else:
            neg_scale_adjust, pos_scale_adjust = strength
        
        for layer_name in lrm_layer_names:
            if 'affective_valence_128d' in layer_name:
                # Convert activation to 128D valence representation
                valence_128d = self._convert_activation_to_128d(activation, 'valence')
                # Convert to 4D for conv layer modulation
                valence_4d = valence_128d.unsqueeze(-1).unsqueeze(-1)  # (batch, 128, 1, 1)
                
                # Store steering signals in the correct LRM module locations
                # Find all ModBlocks that use this source
                pattern = 'from_affective_valence_128d_to_'
                for lrm_module_name, lrm_module in self.lrm.named_children():
                    for mod_name, mod_module in lrm_module.named_children():
                        if pattern in mod_name:
                            # Apply strength adjustment to the ModBlock parameters
                            self.adjust_modulation_strengths(mod_module, neg_scale_adjust, pos_scale_adjust)
                            
                            if mod_name in lrm_module.mod_inputs:
                                # Blend with existing signal
                                curr_input = lrm_module.mod_inputs[mod_name]
                                lrm_module.mod_inputs[mod_name] = alpha * valence_4d + (1 - alpha) * curr_input
                            else:
                                # Set new steering signal
                                lrm_module.mod_inputs[mod_name] = valence_4d
                
            elif 'affective_arousal_128d' in layer_name:
                # Convert activation to 128D arousal representation
                arousal_128d = self._convert_activation_to_128d(activation, 'arousal')
                # Convert to 4D for conv layer modulation
                arousal_4d = arousal_128d.unsqueeze(-1).unsqueeze(-1)  # (batch, 128, 1, 1)
                
                # Store steering signals in the correct LRM module locations
                # Find all ModBlocks that use this source
                pattern = 'from_affective_arousal_128d_to_'
                for lrm_module_name, lrm_module in self.lrm.named_children():
                    for mod_name, mod_module in lrm_module.named_children():
                        if pattern in mod_name:
                            # Apply strength adjustment to the ModBlock parameters
                            self.adjust_modulation_strengths(mod_module, neg_scale_adjust, pos_scale_adjust)
                            
                            if mod_name in lrm_module.mod_inputs:
                                # Blend with existing signal
                                curr_input = lrm_module.mod_inputs[mod_name]
                                lrm_module.mod_inputs[mod_name] = alpha * arousal_4d + (1 - alpha) * curr_input
                            else:
                                # Set new steering signal
                                lrm_module.mod_inputs[mod_name] = arousal_4d
    
    def _convert_activation_to_128d(self, activation, emotion_type):
        """
        Convert external activation to 128D emotion representation.
        
        Args:
            activation: external activation tensor
            emotion_type: 'valence' or 'arousal'
        
        Returns:
            128D tensor compatible with our affective pathways
        """
        # Handle different activation shapes
        if activation.dim() == 1:
            # 1D activation (e.g., classifier output)
            if activation.shape[0] == 128:
                # Already 128D
                return activation.unsqueeze(0)  # Add batch dimension
            else:
                # Project to 128D using a simple linear transformation
                # For now, use a simple approach - repeat or truncate
                if activation.shape[0] > 128:
                    return activation[:128].unsqueeze(0)
                else:
                    # Pad with zeros
                    padded = torch.zeros(128, device=activation.device, dtype=activation.dtype)
                    padded[:activation.shape[0]] = activation
                    return padded.unsqueeze(0)
        elif activation.dim() == 2:
            # 2D activation (batch, features)
            batch_size = activation.shape[0]
            if activation.shape[1] == 128:
                return activation
            elif activation.shape[1] > 128:
                return activation[:, :128]
            else:
                # Pad with zeros
                padded = torch.zeros(batch_size, 128, device=activation.device, dtype=activation.dtype)
                padded[:, :activation.shape[1]] = activation
                return padded
        else:
            # Higher dimensional - flatten and convert
            flattened = activation.flatten(start_dim=1)
            return self._convert_activation_to_128d(flattened, emotion_type)
    
    def _set_active_connections(self, active_connections):
        """
        Set specific active connections for LRM.
        
        Args:
            active_connections: list of tuples [('source_layer', 'target_layer'), ...]
        """
        # Store original connections
        if not hasattr(self, '_original_mod_connections'):
            self._original_mod_connections = self.lrm.mod_connections.copy()
        
        # Filter connections based on active_connections
        if active_connections:
            filtered_connections = []
            for conn in self.lrm.mod_connections:
                source = conn['source']
                target = conn['destination']
                
                # Check if this connection should be active
                for active_source, active_target in active_connections:
                    if (active_source in source or source in active_source) and \
                       (active_target in target or target in active_target):
                        filtered_connections.append(conn)
                        break
            
            self.lrm.mod_connections = filtered_connections
    
    def _reset_active_connections(self):
        """Reset to original connections."""
        if hasattr(self, '_original_mod_connections'):
            self.lrm.mod_connections = self._original_mod_connections
    
    def extract_activations(self, input):
        """
        Extract intermediate activations from the model for steering signal generation.
        
        Args:
            input: (batch_size, time_steps, mel_bins) pre-computed features
            
        Returns:
            activations: dict containing intermediate activations
        """
        # Handle input dimensions
        if input.dim() == 4 and input.shape[1] == 1:
            input = input.squeeze(1)  # (batch_size, time_steps, mel_bins)
        
        # Forward through visual system
        visual_embedding = self._forward_visual_system(input, mixup_lambda=None)
        
        # Extract activations from affective pathways
        valence_256d = self.affective_valence[0:2](visual_embedding)  # Linear(512,256) + ReLU
        valence_128d = self.affective_valence[2:4](valence_256d)      # Linear(256,128) + ReLU
        valence_out = self.affective_valence[4](valence_128d)         # Linear(128,1)
        
        arousal_256d = self.affective_arousal[0:2](visual_embedding)  # Linear(512,256) + ReLU
        arousal_128d = self.affective_arousal[2:4](arousal_256d)      # Linear(256,128) + ReLU
        arousal_out = self.affective_arousal[4](arousal_128d)         # Linear(128,1)
        
        # Return all activations
        activations = {
            'valence_128d': valence_128d,
            'arousal_128d': arousal_128d,
            'valence_256d': valence_256d,
            'arousal_256d': arousal_256d,
            'visual_embedding': visual_embedding,
            'valence_output': valence_out,
            'arousal_output': arousal_out
        }
        
        return activations 

    def add_steering_signal(self, source, activation, strength, alpha=1):
        """
        Add a steering signal to the network (compatible with SteerableLRM interface).
        Parameters
        ----------
        source: source layer name
        activation: steering signal
        strength: modulation strength
        alpha: the ratio of the steering signal to the current source layer's output
        Returns
        -------
        None
        """
        print(f"    🎯 add_steering_signal called with source='{source}', activation.shape={activation.shape}")
        
        # Handle strength parameter (can be float or tuple)
        if isinstance(strength, (int, float)):
            neg_scale_adjust, pos_scale_adjust = strength, strength
        else:
            neg_scale_adjust, pos_scale_adjust = strength

        # find modules targeted by this steering_source, replace their
        # modulatory inputs for this source with steering activation
        # Convert source name to match the module naming convention
        source_clean = source.replace(".", "_")
        pattern = f'from_{source_clean}_to_'
        
        # Debug: Print what we're looking for
        print(f"      Looking for pattern: '{pattern}'")
        print(f"      Available feedback modules:")
        
        found_matching_module = False
        for lrm_module_name, lrm_module in self.lrm.named_children():
            for feedback_module_name, feedback_module in lrm_module.named_children():
                print(f"        {feedback_module_name}")
                # If this ModBlock is from steering source to current lrm_module, add steering
                if pattern in feedback_module_name:
                    found_matching_module = True
                    # adjust modulation strengths
                    self.adjust_modulation_strengths(feedback_module, neg_scale_adjust, pos_scale_adjust)

                    # Align the shape of the activation to match curr_input
                    if activation.dim() == 1:
                        activation = activation.unsqueeze(0)  # Convert [1000] to [1, 1000] when first_pass_steering is True
                    elif activation.dim() == 2:
                        activation = activation.unsqueeze(-1).unsqueeze(-1)  # Convert [batch, 128] to [batch, 128, 1, 1] for conv modulation
                    elif activation.dim() == 3:
                        activation = activation.unsqueeze(0) # Convert [256, 6, 6] to [1, 256, 6, 6] when first_pass_steering is True

                    # Store the steering signal in the main mod_inputs dictionary
                    # This is what the forward hook looks for
                    print(f"      Storing steering signal in {lrm_module_name}.mod_inputs[{feedback_module_name}]")
                    if feedback_module_name in lrm_module.mod_inputs:
                        # update current modulatory input (if alpha==1, we end up just replacing it with steering activation)
                        # the activity of the source layer in the recent pass; it is saved by hook_fn function in layers.py
                        curr_input = lrm_module.mod_inputs[feedback_module_name]
                        # the feedback signal is a combination of the external steering signal and the current source layer's output
                        lrm_module.mod_inputs[feedback_module_name] = alpha * activation + (1 - alpha) * curr_input
                        print(f"        Updated existing signal, new shape: {lrm_module.mod_inputs[feedback_module_name].shape}")
                else:
                        # without curr_input, simply set to template
                        lrm_module.mod_inputs[feedback_module_name] = activation
                        print(f"        Set new signal, shape: {lrm_module.mod_inputs[feedback_module_name].shape}")
                        
        if not found_matching_module:
            print(f"      WARNING: No matching module found for pattern '{pattern}'")

    @torch.no_grad()
    def adjust_modulation_strengths(self, feedback_module, neg_scale_adjust, pos_scale_adjust):
        """Adjust modulation strengths for a specific feedback module."""
        if neg_scale_adjust != 1.0:
            if not hasattr(feedback_module, 'neg_scale_orig'):
                feedback_module.neg_scale_orig = feedback_module.neg_scale.clone()
            feedback_module.neg_scale = nn.Parameter(feedback_module.neg_scale_orig * neg_scale_adjust)

        if pos_scale_adjust != 1.0:
            if not hasattr(feedback_module, 'pos_scale_orig'):
                feedback_module.pos_scale_orig = feedback_module.pos_scale.clone()
            feedback_module.pos_scale = nn.Parameter(feedback_module.pos_scale_orig * pos_scale_adjust)

    def add_steering_signals(self, steering_signals):
        """
        Add multiple steering signals to the network.
        
        Parameters
        ----------
        steering_signals: list of steering signal dictionaries
        """
        for steering_signal in steering_signals:
            self.add_steering_signal(**steering_signal)

    def reset_modulation_strengths(self):
        """
        Reset modulation strengths to original values.
        """
        for lrm_module_name, lrm_module in self.lrm.named_children():
            for feedback_module_name, feedback_module in lrm_module.named_children():
                if hasattr(feedback_module, 'neg_scale_orig'):
                    feedback_module.neg_scale = nn.Parameter(feedback_module.neg_scale_orig)
                    del feedback_module.neg_scale_orig
                if hasattr(feedback_module, 'pos_scale_orig'):
                    feedback_module.pos_scale = nn.Parameter(feedback_module.pos_scale_orig)
                    del feedback_module.pos_scale_orig 