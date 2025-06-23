#!/usr/bin/env python3
"""
Loss functions for emotion regression training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_func(loss_type):
    """Get loss function based on loss type.
    
    Args:
        loss_type: str, type of loss function ('mse', 'mae', 'smooth_l1')
        
    Returns:
        loss_func: function that takes (output_dict, target_dict) and returns loss
    """
    if loss_type == 'mse':
        return mse_loss
    elif loss_type == 'mae':
        return mae_loss
    elif loss_type == 'smooth_l1':
        return smooth_l1_loss
    else:
        raise ValueError(f'Unknown loss type: {loss_type}')


def mse_loss(output_dict, target_dict):
    """Mean squared error loss for emotion regression.
    
    Args:
        output_dict: dict containing 'valence' and 'arousal' predictions
        target_dict: dict containing 'valence' and 'arousal' targets
        
    Returns:
        loss: torch.Tensor, total loss
    """
    # Ensure targets have the same shape as predictions
    valence_target = target_dict['valence'].unsqueeze(-1) if target_dict['valence'].dim() == 1 else target_dict['valence']
    arousal_target = target_dict['arousal'].unsqueeze(-1) if target_dict['arousal'].dim() == 1 else target_dict['arousal']
    
    valence_loss = F.mse_loss(output_dict['valence'], valence_target)
    arousal_loss = F.mse_loss(output_dict['arousal'], arousal_target)
    return valence_loss + arousal_loss


def mae_loss(output_dict, target_dict):
    """Mean absolute error loss for emotion regression.
    
    Args:
        output_dict: dict containing 'valence' and 'arousal' predictions
        target_dict: dict containing 'valence' and 'arousal' targets
        
    Returns:
        loss: torch.Tensor, total loss
    """
    # Ensure targets have the same shape as predictions
    valence_target = target_dict['valence'].unsqueeze(-1) if target_dict['valence'].dim() == 1 else target_dict['valence']
    arousal_target = target_dict['arousal'].unsqueeze(-1) if target_dict['arousal'].dim() == 1 else target_dict['arousal']
    
    valence_loss = F.l1_loss(output_dict['valence'], valence_target)
    arousal_loss = F.l1_loss(output_dict['arousal'], arousal_target)
    return valence_loss + arousal_loss


def smooth_l1_loss(output_dict, target_dict):
    """Smooth L1 loss for emotion regression.
    
    Args:
        output_dict: dict containing 'valence' and 'arousal' predictions
        target_dict: dict containing 'valence' and 'arousal' targets
        
    Returns:
        loss: torch.Tensor, total loss
    """
    # Ensure targets have the same shape as predictions
    valence_target = target_dict['valence'].unsqueeze(-1) if target_dict['valence'].dim() == 1 else target_dict['valence']
    arousal_target = target_dict['arousal'].unsqueeze(-1) if target_dict['arousal'].dim() == 1 else target_dict['arousal']
    
    valence_loss = F.smooth_l1_loss(output_dict['valence'], valence_target)
    arousal_loss = F.smooth_l1_loss(output_dict['arousal'], arousal_target)
    return valence_loss + arousal_loss 