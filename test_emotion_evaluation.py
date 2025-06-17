#!/usr/bin/env python3

import numpy as np
import h5py
import sys
import os

# Add pytorch directory to path
sys.path.append('pytorch')

from pytorch.emotion_evaluate import EmotionEvaluator
from pytorch.data_generator import EmoSoundscapesDataset, EmotionValidateSampler, emotion_collate_fn
import torch

def test_emotion_evaluation(feature_path):
    """Test the emotion evaluation with both segment and audio-level metrics."""
    
    print("Testing Emotion Evaluation...")
    print(f"Feature file: {feature_path}")
    
    # Check if feature file exists
    if not os.path.exists(feature_path):
        print(f"⚠️  Feature file {feature_path} not found (trying alternative locations)")
        return
    
    # Load feature file to check structure
    with h5py.File(feature_path, 'r') as hf:
        print(f"Feature shape: {hf['feature'].shape}")
        print(f"Audio names: {len(hf['audio_name'])}")
        print(f"Valence range: [{np.min(hf['valence'][:]):.3f}, {np.max(hf['valence'][:]):.3f}]")
        print(f"Arousal range: [{np.min(hf['arousal'][:]):.3f}, {np.max(hf['arousal'][:]):.3f}]")
        
        # Check segment naming
        sample_names = [name.decode() if isinstance(name, bytes) else name 
                       for name in hf['audio_name'][:10]]
        print(f"Sample audio names: {sample_names}")
        
        # Count unique audio files and check for segmentation
        base_names = set()
        segmented_count = 0
        for name in hf['audio_name'][:]:
            name_str = name.decode() if isinstance(name, bytes) else name
            if '_seg' in name_str:
                segmented_count += 1
            base_name = name_str.split('_seg')[0] if '_seg' in name_str else name_str
            base_names.add(base_name)
        
        print(f"Unique audio files: {len(base_names)}")
        print(f"Segmented samples: {segmented_count}/{len(hf['audio_name'])}")
        
        if segmented_count == 0:
            print("⚠️  WARNING: Features appear to be in OLD FORMAT (not segmented)")
            print("   You need to re-extract features with the updated extract_emotion_features.py")
            print("   Expected: 6 segments per audio file (1213 files → 7278 segments)")
            print("   Current: 1 sample per audio file (1213 files → 1213 samples)")
        else:
            print("✅ Features are in segmented format")
    
    # Create a simple mock model for testing
    class MockEmotionModel:
        def __init__(self):
            self.training = False
            
        def eval(self):
            pass
            
        def __call__(self, features, targets):
            # Return random predictions for testing
            batch_size = features.shape[0]
            return {
                'valence': torch.randn(batch_size, 1) * 0.5,  # Random predictions
                'arousal': torch.randn(batch_size, 1) * 0.5
            }
    
    # Create dataset and data loader
    dataset = EmoSoundscapesDataset()
    val_sampler = EmotionValidateSampler(feature_path, batch_size=32, train_ratio=0.7)
    
    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=val_sampler,
        collate_fn=emotion_collate_fn,
        num_workers=0,  # Use 0 for testing
        pin_memory=False
    )
    
    # Check if we have enough validation samples
    if len(val_sampler.val_indices) == 0:
        print("Error: No validation samples found!")
        return
    
    print(f"\nValidation samples: {len(val_sampler.val_indices)}")
    print(f"Expected batches: ~{len(val_sampler.val_indices) // 32}")
    
    # Test evaluation
    mock_model = MockEmotionModel()
    evaluator = EmotionEvaluator(mock_model)
    
    print("\nRunning evaluation...")
    statistics = evaluator.evaluate(val_loader)
    
    print("\nEvaluation Results:")
    evaluator.print_evaluation(statistics)
    
    # Verify that we have both segment and audio level metrics
    expected_keys = [
        'segment_mean_mae', 'segment_mean_rmse', 'segment_mean_pearson',
        'audio_mean_mae', 'audio_mean_rmse', 'audio_mean_pearson',
        'segment_num_samples', 'audio_num_samples'
    ]
    
    print(f"\nMetric verification:")
    for key in expected_keys:
        if key in statistics:
            print(f"✓ {key}: {statistics[key]}")
        else:
            print(f"✗ Missing: {key}")
    
    # Check that audio-level has fewer samples than segment-level
    if statistics['audio_num_samples'] < statistics['segment_num_samples']:
        print(f"✓ Audio-level aggregation working: {statistics['audio_num_samples']} audio files < {statistics['segment_num_samples']} segments")
    else:
        print(f"✗ Audio-level aggregation issue: {statistics['audio_num_samples']} audio files >= {statistics['segment_num_samples']} segments")

if __name__ == '__main__':
    # Test with the emotion features - try multiple locations
    possible_paths = [
        'workspaces/emotion_regression/features/emotion_features.h5',
        'workspaces/emotion_feedback/features/emotion_features.h5', 
        'features/emotion_features/emotion_features.h5',
        'emotion_features.h5'
    ]
    
    feature_path = None
    for path in possible_paths:
        if os.path.exists(path):
            feature_path = path
            print(f"✅ Found emotion features at: {path}")
            break
        else:
            print(f"⚠️  Checking {path}... not found")
    
    if feature_path is None:
        print("❌ No emotion features found in any expected location!")
        print("Expected locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease run feature extraction first:")
        print("  bash run_emotion.sh")
        sys.exit(1)
    
    test_emotion_evaluation(feature_path) 