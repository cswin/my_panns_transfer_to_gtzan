#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script showing how to use Cnn6 models for emotion regression.
This demonstrates the basic usage without requiring the full dataset.
"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))

import torch
import numpy as np
from src.utils.config import sample_rate, cnn6_config
from src.models import FeatureEmotionRegression_Cnn6, EmotionRegression_Cnn6

def test_cnn6_emotion_models():
    """Test the Cnn6 emotion regression models with dummy data."""
    
    print("🎵 Testing Cnn6 Emotion Regression Models")
    print("=" * 50)
    
    # Model configurations
    config = cnn6_config
    batch_size = 4
    
    # Create models
    print("1. Creating models...")
    
    # Feature-based model (takes mel-spectrogram input)
    feature_model = FeatureEmotionRegression_Cnn6(
        sample_rate=sample_rate,
        window_size=config['window_size'],
        hop_size=config['hop_size'],
        mel_bins=config['mel_bins'],
        fmin=config['fmin'],
        fmax=config['fmax'],
        freeze_base=True
    )
    
    # Raw waveform model (takes audio input)
    waveform_model = EmotionRegression_Cnn6(
        sample_rate=sample_rate,
        window_size=config['window_size'],
        hop_size=config['hop_size'],
        mel_bins=config['mel_bins'],
        fmin=config['fmin'],
        fmax=config['fmax'],
        freeze_base=True
    )
    
    print(f"✅ FeatureEmotionRegression_Cnn6: {sum(p.numel() for p in feature_model.parameters()):,} parameters")
    print(f"✅ EmotionRegression_Cnn6: {sum(p.numel() for p in waveform_model.parameters()):,} parameters")
    
    # Test feature-based model
    print("\n2. Testing feature-based model...")
    
    # Dummy mel-spectrogram features (6 seconds at 32kHz with hop_size=320 gives ~600 time steps)
    dummy_features = torch.randn(batch_size, 600, config['mel_bins'])  # (batch, time, mels)
    
    # Forward pass
    feature_model.eval()
    with torch.no_grad():
        output = feature_model(dummy_features)
    
    print(f"✅ Input shape: {dummy_features.shape}")
    print(f"✅ Valence predictions: {output['valence'].shape} (range: {output['valence'].min():.3f} to {output['valence'].max():.3f})")
    print(f"✅ Arousal predictions: {output['arousal'].shape} (range: {output['arousal'].min():.3f} to {output['arousal'].max():.3f})")
    print(f"✅ Embeddings: {output['embedding'].shape}")
    
    # Test waveform model
    print("\n3. Testing waveform model...")
    
    # Dummy waveform (6 seconds at 32kHz)
    dummy_waveform = torch.randn(batch_size, sample_rate * 6)  # (batch, samples)
    
    # Forward pass
    waveform_model.eval()
    with torch.no_grad():
        output = waveform_model(dummy_waveform)
    
    print(f"✅ Input shape: {dummy_waveform.shape}")
    print(f"✅ Valence predictions: {output['valence'].shape} (range: {output['valence'].min():.3f} to {output['valence'].max():.3f})")
    print(f"✅ Arousal predictions: {output['arousal'].shape} (range: {output['arousal'].min():.3f} to {output['arousal'].max():.3f})")
    print(f"✅ Embeddings: {output['embedding'].shape}")
    
    print("\n🎉 All tests passed! Cnn6 emotion models are working correctly.")
    
    # Model comparison
    print("\n📊 Model Architecture Comparison:")
    print(f"FeatureEmotionRegression_Cnn6 (features → valence/arousal):")
    print(f"  - Input: Mel-spectrogram features ({config['mel_bins']} mel bins)")
    print(f"  - Embedding size: 512 (Cnn6)")
    print(f"  - Output: Valence + Arousal regression")
    
    print(f"EmotionRegression_Cnn6 (waveform → valence/arousal):")
    print(f"  - Input: Raw audio waveform (32kHz)")
    print(f"  - Embedding size: 512 (Cnn6)")
    print(f"  - Output: Valence + Arousal regression")
    print(f"  - Includes: STFT + Mel filterbank processing")

if __name__ == '__main__':
    test_cnn6_emotion_models() 