import torch
import torch.nn.functional as F


def clip_nll(output_dict, target_dict):
    loss = - torch.mean(target_dict['target'] * output_dict['clipwise_output'])
    return loss


def mse_loss(output_dict, target_dict):
    """Mean Squared Error loss for regression tasks."""
    valence_loss = F.mse_loss(output_dict['valence'].squeeze(), target_dict['valence'])
    arousal_loss = F.mse_loss(output_dict['arousal'].squeeze(), target_dict['arousal'])
    return valence_loss + arousal_loss


def mae_loss(output_dict, target_dict):
    """Mean Absolute Error (L1) loss for regression tasks."""
    valence_loss = F.l1_loss(output_dict['valence'].squeeze(), target_dict['valence'])
    arousal_loss = F.l1_loss(output_dict['arousal'].squeeze(), target_dict['arousal'])
    return valence_loss + arousal_loss


def smooth_l1_loss(output_dict, target_dict):
    """Smooth L1 (Huber) loss for regression tasks."""
    valence_loss = F.smooth_l1_loss(output_dict['valence'].squeeze(), target_dict['valence'])
    arousal_loss = F.smooth_l1_loss(output_dict['arousal'].squeeze(), target_dict['arousal'])
    return valence_loss + arousal_loss


def weighted_mse_loss(output_dict, target_dict, valence_weight=1.0, arousal_weight=1.0):
    """Weighted MSE loss for different emphasis on valence vs arousal."""
    valence_loss = F.mse_loss(output_dict['valence'].squeeze(), target_dict['valence'])
    arousal_loss = F.mse_loss(output_dict['arousal'].squeeze(), target_dict['arousal'])
    return valence_weight * valence_loss + arousal_weight * arousal_loss


def get_loss_func(loss_type):
    if loss_type == 'clip_nll':
        return clip_nll
    elif loss_type == 'mse':
        return mse_loss
    elif loss_type == 'mae':
        return mae_loss
    elif loss_type == 'smooth_l1':
        return smooth_l1_loss
    elif loss_type == 'weighted_mse':
        return weighted_mse_loss
    else:
        raise ValueError(f'Unknown loss type: {loss_type}')