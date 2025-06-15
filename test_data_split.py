#!/usr/bin/env python3
"""
Test script to verify audio-based data splitting prevents data leakage.
"""

import h5py
import numpy as np
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))

from data_generator import EmotionTrainSampler, EmotionValidateSampler

def test_data_split(feature_path, train_ratio=0.7):
    """Test the data splitting to ensure no audio file appears in both train and validation."""
    
    print("Testing Audio-Based Data Split")
    print("=" * 60)
    
    # Load features
    with h5py.File(feature_path, 'r') as hf:
        audio_names = [name.decode() if isinstance(name, bytes) else name for name in hf['audio_name'][:]]
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
    
    print(f"Total samples: {len(audio_names)}")
    
    # Create samplers
    train_sampler = EmotionTrainSampler(feature_path, batch_size=32, train_ratio=train_ratio)
    val_sampler = EmotionValidateSampler(feature_path, batch_size=32, train_ratio=train_ratio)
    
    print(f"\nTrain indices: {len(train_sampler.train_indices)}")
    print(f"Val indices: {len(val_sampler.val_indices)}")
    
    # Get audio names for each split
    train_audio_names = [audio_names[i] for i in train_sampler.train_indices]
    val_audio_names = [audio_names[i] for i in val_sampler.val_indices]
    
    # Extract base audio file names (remove segment suffixes)
    def get_base_name(name):
        return name.split('_seg')[0] if '_seg' in name else name
    
    train_base_files = set([get_base_name(name) for name in train_audio_names])
    val_base_files = set([get_base_name(name) for name in val_audio_names])
    
    print(f"\nUnique audio files in train: {len(train_base_files)}")
    print(f"Unique audio files in val: {len(val_base_files)}")
    
    # Check for overlap (data leakage)
    overlap = train_base_files.intersection(val_base_files)
    
    if len(overlap) == 0:
        print("✅ NO DATA LEAKAGE: No audio files appear in both train and validation sets")
    else:
        print(f"❌ DATA LEAKAGE DETECTED: {len(overlap)} audio files appear in both sets:")
        for file in sorted(overlap):
            print(f"  - {file}")
    
    # Show distribution of segments per audio file
    print(f"\nSegment distribution:")
    
    # Count segments per base file in train set
    train_segment_counts = {}
    for name in train_audio_names:
        base_name = get_base_name(name)
        train_segment_counts[base_name] = train_segment_counts.get(base_name, 0) + 1
    
    # Count segments per base file in val set
    val_segment_counts = {}
    for name in val_audio_names:
        base_name = get_base_name(name)
        val_segment_counts[base_name] = val_segment_counts.get(base_name, 0) + 1
    
    print(f"Train set - segments per audio file: {list(train_segment_counts.values())[:10]}...")
    print(f"Val set - segments per audio file: {list(val_segment_counts.values())[:10]}...")
    
    # Expected: 6 segments per audio file (since we split 6-second clips into 1-second segments)
    expected_segments = 6
    train_correct_segments = sum(1 for count in train_segment_counts.values() if count == expected_segments)
    val_correct_segments = sum(1 for count in val_segment_counts.values() if count == expected_segments)
    
    print(f"\nFiles with expected {expected_segments} segments:")
    print(f"  Train: {train_correct_segments}/{len(train_segment_counts)} ({train_correct_segments/len(train_segment_counts)*100:.1f}%)")
    print(f"  Val: {val_correct_segments}/{len(val_segment_counts)} ({val_correct_segments/len(val_segment_counts)*100:.1f}%)")
    
    # Show some examples
    print(f"\nExample train files and their segments:")
    train_examples = sorted(list(train_base_files))[:3]
    for base_file in train_examples:
        segments = [name for name in train_audio_names if get_base_name(name) == base_file]
        print(f"  {base_file}: {len(segments)} segments")
        for seg in segments[:3]:  # Show first 3 segments
            idx = train_sampler.train_indices[train_audio_names.index(seg)]
            print(f"    {seg} -> valence={valence[idx]:.3f}, arousal={arousal[idx]:.3f}")
    
    print(f"\nExample val files and their segments:")
    val_examples = sorted(list(val_base_files))[:3]
    for base_file in val_examples:
        segments = [name for name in val_audio_names if get_base_name(name) == base_file]
        print(f"  {base_file}: {len(segments)} segments")
        for seg in segments[:3]:  # Show first 3 segments
            idx = val_sampler.val_indices[val_audio_names.index(seg)]
            print(f"    {seg} -> valence={valence[idx]:.3f}, arousal={arousal[idx]:.3f}")
    
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