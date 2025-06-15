#!/usr/bin/env python3
"""
Test script to verify emotion feature extraction produces GTZAN-compatible features.
"""

import h5py
import numpy as np

def test_emotion_features(feature_path):
    """Test the extracted emotion features."""
    
    print("Testing emotion features...")
    print("=" * 50)
    
    # Load features
    with h5py.File(feature_path, 'r') as hf:
        features = hf['feature'][:]
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        audio_names = hf['audio_name'][:]
    
    print(f"Feature shape: {features.shape}")
    print(f"Valence shape: {valence.shape}")
    print(f"Arousal shape: {arousal.shape}")
    print(f"Audio names shape: {audio_names.shape}")
    
    # Expected: Each 6-second clip should produce 6 segments
    # So 1213 clips * 6 segments = 7278 total samples
    expected_samples = 1213 * 6
    
    print(f"\nExpected samples: {expected_samples}")
    print(f"Actual samples: {features.shape[0]}")
    
    # Check feature dimensions (should be similar to GTZAN)
    print(f"\nFeature dimensions:")
    print(f"Time steps: {features.shape[1]} (should be ~101 for 1-second segments)")
    print(f"Mel bins: {features.shape[2]} (should be 64)")
    
    # Check rating ranges
    print(f"\nRating ranges:")
    print(f"Valence: [{valence.min():.3f}, {valence.max():.3f}]")
    print(f"Arousal: [{arousal.min():.3f}, {arousal.max():.3f}]")
    
    # Show some sample names
    print(f"\nSample audio names:")
    for i in range(min(10, len(audio_names))):
        name = audio_names[i].decode() if isinstance(audio_names[i], bytes) else audio_names[i]
        print(f"  {i}: {name}")
    
    # Check if segments from same file have same ratings
    print(f"\nChecking segment consistency...")
    sample_base_name = audio_names[0].decode().split('_seg')[0] if isinstance(audio_names[0], bytes) else audio_names[0].split('_seg')[0]
    
    # Find all segments from the same original file
    same_file_indices = []
    for i, name in enumerate(audio_names):
        name_str = name.decode() if isinstance(name, bytes) else name
        if name_str.startswith(sample_base_name):
            same_file_indices.append(i)
    
    print(f"Segments from '{sample_base_name}':")
    for idx in same_file_indices:
        name = audio_names[idx].decode() if isinstance(audio_names[idx], bytes) else audio_names[idx]
        print(f"  {name}: valence={valence[idx]:.3f}, arousal={arousal[idx]:.3f}")
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python test_emotion_features.py <path_to_emotion_features.h5>")
        sys.exit(1)
    
    feature_path = sys.argv[1]
    test_emotion_features(feature_path) 