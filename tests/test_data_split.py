#!/usr/bin/env python3
"""
Test script to verify audio-based data splitting prevents data leakage.
"""

import h5py
import numpy as np
import sys
import os

from src.data.data_generator import EmotionTrainSampler, EmotionValidateSampler

def test_data_split(feature_path, train_ratio=0.7):
    """Test the data splitting to ensure no audio file appears in both train and validation."""
    
    print("Testing Audio-Based Data Split (Full-Length Audios)")
    print("=" * 60)
    
    # Load features and store important info before closing
    with h5py.File(feature_path, 'r') as hf:
        audio_names = [name.decode() if isinstance(name, bytes) else name for name in hf['audio_name'][:]]
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        feature_shape = hf['feature'].shape  # Store shape before closing
    
    print(f"Total audio files: {len(audio_names)}")
    
    # Create samplers
    train_sampler = EmotionTrainSampler(feature_path, batch_size=32, train_ratio=train_ratio)
    val_sampler = EmotionValidateSampler(feature_path, batch_size=32, train_ratio=train_ratio)
    
    print(f"\nTrain indices: {len(train_sampler.train_indexes)}")
    print(f"Val indices: {len(val_sampler.validate_indexes)}")
    
    # Get audio names for each split
    train_audio_names = [audio_names[i] for i in train_sampler.train_indexes]
    val_audio_names = [audio_names[i] for i in val_sampler.validate_indexes]
    
    print(f"\nUnique audio files in train: {len(train_audio_names)}")
    print(f"Unique audio files in val: {len(val_audio_names)}")
    
    # Check for overlap (data leakage)
    train_set = set(train_audio_names)
    val_set = set(val_audio_names)
    overlap = train_set.intersection(val_set)
    
    if len(overlap) == 0:
        print("✅ NO DATA LEAKAGE: No audio files appear in both train and validation sets")
    else:
        print(f"❌ DATA LEAKAGE DETECTED: {len(overlap)} audio files appear in both sets:")
        for file in sorted(overlap):
            print(f"  - {file}")
    
    # Show distribution of audio files
    print(f"\nAudio file distribution:")
    print(f"  Train set: {len(train_audio_names)} files ({len(train_audio_names)/len(audio_names)*100:.1f}%)")
    print(f"  Val set: {len(val_audio_names)} files ({len(val_audio_names)/len(audio_names)*100:.1f}%)")
    
    # Show some examples
    print(f"\nExample train files:")
    train_examples = train_audio_names[:3]
    for audio_file in train_examples:
        idx = train_sampler.train_indexes[train_audio_names.index(audio_file)]
        print(f"  {audio_file} -> valence={valence[idx]:.3f}, arousal={arousal[idx]:.3f}")
    
    print(f"\nExample val files:")
    val_examples = val_audio_names[:3]
    for audio_file in val_examples:
        idx = val_sampler.validate_indexes[val_audio_names.index(audio_file)]
        print(f"  {audio_file} -> valence={valence[idx]:.3f}, arousal={arousal[idx]:.3f}")
    
    # Show feature statistics
    print(f"\nFeature statistics:")
    print(f"  Feature shape: {feature_shape}")
    print(f"  Valence range: [{valence.min():.3f}, {valence.max():.3f}]")
    print(f"  Arousal range: [{arousal.min():.3f}, {arousal.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("Data split test completed!")
    
    return len(overlap) == 0  # Return True if no data leakage

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_data_split.py <path_to_emotion_features.h5>")
        sys.exit(1)
    
    feature_path = sys.argv[1]
    
    if not os.path.exists(feature_path):
        print(f"Error: Feature file not found: {feature_path}")
        sys.exit(1)
    
    success = test_data_split(feature_path)
    sys.exit(0 if success else 1) 