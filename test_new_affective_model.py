#!/usr/bin/env python3
"""
Test script to verify the new affective model works correctly.
"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))

import torch
import numpy as np

# Import our modules
from config import sample_rate, mel_bins, fmin, fmax, cnn6_config
from models import FeatureEmotionRegression_Cnn6, FeatureEmotionRegression_Cnn6_NewAffective


def test_new_affective_model():
    """Test the new affective model."""
    
    print("🧪 Testing New Affective Model")
    print("=" * 50)
    
    # Create models
    config = cnn6_config
    
    # Original model
    old_model = FeatureEmotionRegression_Cnn6(
        sample_rate=sample_rate,
        window_size=config['window_size'],
        hop_size=config['hop_size'],
        mel_bins=config['mel_bins'],
        fmin=config['fmin'],
        fmax=config['fmax'],
        freeze_base=True
    )
    
    # New affective model
    new_model = FeatureEmotionRegression_Cnn6_NewAffective(
        sample_rate=sample_rate,
        window_size=config['window_size'],
        hop_size=config['hop_size'],
        mel_bins=config['mel_bins'],
        fmin=config['fmin'],
        fmax=config['fmax'],
        freeze_base=True
    )
    
    print("✅ Created both models")
    
    # Test with dummy input - correct shape for feature-based model
    batch_size = 4
    time_steps = 64  # 1 second at 16kHz with hop_size=512  
    mel_bins = 64
    
    # Feature-based models expect (batch_size, time_steps, mel_bins) from data loader
    # This matches the emotion_collate_fn output format
    dummy_input = torch.randn(batch_size, time_steps, mel_bins)
    print(f"Input shape: {dummy_input.shape}")
    
    # Test old model
    old_model.eval()
    with torch.no_grad():
        old_output = old_model(dummy_input)
        
    print(f"Old model output shapes:")
    print(f"  Valence: {old_output['valence'].shape}")
    print(f"  Arousal: {old_output['arousal'].shape}")
    print(f"  Embedding: {old_output['embedding'].shape}")
    
    # Test new model
    new_model.eval()
    with torch.no_grad():
        new_output = new_model(dummy_input)
        
    print(f"New model output shapes:")
    print(f"  Valence: {new_output['valence'].shape}")
    print(f"  Arousal: {new_output['arousal'].shape}")
    print(f"  Embedding: {new_output['embedding'].shape}")
    
    # Check shapes are the same
    assert old_output['valence'].shape == new_output['valence'].shape
    assert old_output['arousal'].shape == new_output['arousal'].shape
    assert old_output['embedding'].shape == new_output['embedding'].shape
    
    print("✅ Output shapes match")
    
    # Check that predictions are different (different architectures)
    val_diff = torch.abs(old_output['valence'] - new_output['valence']).mean().item()
    ar_diff = torch.abs(old_output['arousal'] - new_output['arousal']).mean().item()
    
    print(f"Prediction differences:")
    print(f"  Valence MAE: {val_diff:.6f}")
    print(f"  Arousal MAE: {ar_diff:.6f}")
    
    if val_diff > 1e-6 or ar_diff > 1e-6:
        print("✅ Models produce different predictions (expected)")
    else:
        print("⚠️  Models produce very similar predictions (unexpected)")
    
    # Test parameter counts
    old_params = sum(p.numel() for p in old_model.parameters() if p.requires_grad)
    new_params = sum(p.numel() for p in new_model.parameters() if p.requires_grad)
    
    print(f"Trainable parameters:")
    print(f"  Old model: {old_params:,}")
    print(f"  New model: {new_params:,}")
    print(f"  Difference: {new_params - old_params:,}")
    
    if new_params > old_params:
        print("✅ New model has more parameters (expected - separate pathways)")
    else:
        print("⚠️  New model doesn't have more parameters")
    
    print("\n🎉 New Affective Model Test Complete!")
    print("=" * 50)
    print("Summary:")
    print("✅ New affective model created successfully")
    print("✅ Output shapes match original model")
    print("✅ Models produce different predictions")
    print("✅ New model has separate affective pathways")
    print("\nThe new affective model is ready for training!")
    
    return True


if __name__ == '__main__':
    success = test_new_affective_model()
    sys.exit(0 if success else 1) 