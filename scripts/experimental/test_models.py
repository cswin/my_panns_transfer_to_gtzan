#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model testing script to verify all models work correctly.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.models.cnn_models import Transfer_Cnn6, Transfer_Cnn14, FeatureAffectiveCnn6, EmotionRegression_Cnn6
from src.models.emotion_models import FeatureEmotionRegression_Cnn6
from configs.model_configs import get_model_config, sample_rate

def test_cnn_models():
    """Test CNN-based genre classification models."""
    print("🧪 Testing CNN Genre Classification Models")
    print("=" * 50)
    
    models = {
        'Transfer_Cnn6': Transfer_Cnn6,
        'Transfer_Cnn14': Transfer_Cnn14,
        'FeatureAffectiveCnn6': FeatureAffectiveCnn6,
    }
    
    batch_size = 4
    mel_bins = 64
    time_steps = 600  # ~6 seconds at 32kHz with hop_size=320
    
    for model_name, model_class in models.items():
        print(f"\n📊 Testing {model_name}...")
        
        try:
            # Get model config
            config = get_model_config(model_name)
            
            # Create model
            if model_name == 'FeatureAffectiveCnn6':
                model = model_class(
                    sample_rate=config['sample_rate'],
                    window_size=config['window_size'],
                    hop_size=config['hop_size'],
                    mel_bins=config['mel_bins'],
                    fmin=config['fmin'],
                    fmax=config['fmax'],
                    classes_num=config['classes_num'],
                    freeze_visual_system=True
                )
                # Test with mel-spectrogram input
                dummy_input = torch.randn(batch_size, time_steps, mel_bins)
            else:
                model = model_class(
                    sample_rate=config['sample_rate'],
                    window_size=config['window_size'],
                    hop_size=config['hop_size'],
                    mel_bins=config['mel_bins'],
                    fmin=config['fmin'],
                    fmax=config['fmax'],
                    classes_num=config['classes_num'],
                    freeze_base=True
                )
                # Test with raw waveform input (these models expect waveform)
                dummy_input = torch.randn(batch_size, sample_rate * 6)  # 6 seconds
            
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
                
                # Handle different output formats
                if isinstance(output, dict):
                    # Models that return dictionaries
                    clipwise_output = output['clipwise_output']
                    embedding = output.get('embedding', None)
                    print(f"✅ {model_name}:")
                    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
                    print(f"   - Input shape: {dummy_input.shape}")
                    print(f"   - Output shape: {clipwise_output.shape}")
                    print(f"   - Output range: {clipwise_output.min():.3f} to {clipwise_output.max():.3f}")
                    if embedding is not None:
                        print(f"   - Embedding shape: {embedding.shape}")
                else:
                    # Models that return tensors directly
                    print(f"✅ {model_name}:")
                    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
                    print(f"   - Input shape: {dummy_input.shape}")
                    print(f"   - Output shape: {output.shape}")
                    print(f"   - Output range: {output.min():.3f} to {output.max():.3f}")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")

def test_emotion_models():
    """Test emotion regression models."""
    print("\n🧪 Testing Emotion Regression Models")
    print("=" * 50)
    
    models = {
        'FeatureEmotionRegression_Cnn6': FeatureEmotionRegression_Cnn6,
        'EmotionRegression_Cnn6': EmotionRegression_Cnn6,
    }
    
    batch_size = 4
    mel_bins = 64
    time_steps = 600
    
    for model_name, model_class in models.items():
        print(f"\n📊 Testing {model_name}...")
        
        try:
            # Get model config
            config = get_model_config(model_name)
            
            # Create model
            model = model_class(
                sample_rate=config['sample_rate'],
                window_size=config['window_size'],
                hop_size=config['hop_size'],
                mel_bins=config['mel_bins'],
                fmin=config['fmin'],
                fmax=config['fmax'],
                freeze_base=True
            )
            
            model.eval()
            with torch.no_grad():
                if 'Feature' in model_name:
                    # Test with mel-spectrogram input
                    dummy_input = torch.randn(batch_size, time_steps, mel_bins)
                    output = model(dummy_input)
                    print(f"✅ {model_name} (mel-spectrogram input):")
                else:
                    # Test with raw waveform input
                    dummy_input = torch.randn(batch_size, sample_rate * 6)  # 6 seconds
                    output = model(dummy_input)
                    print(f"✅ {model_name} (waveform input):")
                
                print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
                print(f"   - Input shape: {dummy_input.shape}")
                print(f"   - Valence shape: {output['valence'].shape}")
                print(f"   - Arousal shape: {output['arousal'].shape}")
                print(f"   - Embedding shape: {output['embedding'].shape}")
                print(f"   - Valence range: {output['valence'].min():.3f} to {output['valence'].max():.3f}")
                print(f"   - Arousal range: {output['arousal'].min():.3f} to {output['arousal'].max():.3f}")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")

def test_model_configs():
    """Test model configuration system."""
    print("\n🧪 Testing Model Configuration System")
    print("=" * 50)
    
    model_types = [
        'Transfer_Cnn6',
        'Transfer_Cnn14', 
        'FeatureAffectiveCnn6',
        'FeatureEmotionRegression_Cnn6',
        'EmotionRegression_Cnn6'
    ]
    
    for model_type in model_types:
        try:
            config = get_model_config(model_type)
            print(f"✅ {model_type}: {len(config)} config parameters")
        except Exception as e:
            print(f"❌ {model_type} config failed: {e}")

def main():
    """Main testing function."""
    print("🎵 GTZAN Model Testing Suite")
    print("=" * 50)
    
    # Test model configurations
    test_model_configs()
    
    # Test CNN models
    test_cnn_models()
    
    # Test emotion models
    test_emotion_models()
    
    print("\n🎉 All tests completed!")

if __name__ == '__main__':
    main() 