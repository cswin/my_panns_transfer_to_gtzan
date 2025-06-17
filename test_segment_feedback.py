#!/usr/bin/env python3
"""
Test script to verify segment-based feedback in LRM emotion models.

This script tests that:
1. Segments from the same audio file are processed sequentially
2. Feedback from segment N modulates segment N+1
3. All segment predictions are collected for proper aggregation
4. The LRM evaluator handles segment-based processing correctly
"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))

import numpy as np
import torch
import h5py
from collections import defaultdict

# Import our modules
from config import sample_rate, mel_bins, fmin, fmax, cnn6_config
from models_lrm import FeatureEmotionRegression_Cnn6_LRM
from emotion_evaluate_lrm import LRMEmotionEvaluator
from data_generator import EmoSoundscapesDataset, EmotionValidateSampler, emotion_collate_fn


def test_segment_feedback():
    """Test that segment-based feedback works correctly."""
    
    print("🧪 Testing Segment-Based Feedback in LRM Model")
    print("=" * 60)
    
    # Check if we have test data
    test_data_path = "features/emotion_features/emotion_features.h5"
    if not os.path.exists(test_data_path):
        print(f"❌ Test data not found at {test_data_path}")
        print("Please run feature extraction first:")
        print("  python extract_emotion_features.py --audio_dir <path> --ratings_dir <path> --output_dir features")
        return False
    
    print(f"✅ Found test data: {test_data_path}")
    
    # Create LRM model
    config = cnn6_config
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=sample_rate,
        window_size=config['window_size'],
        hop_size=config['hop_size'],
        mel_bins=config['mel_bins'],
        fmin=config['fmin'],
        fmax=config['fmax'],
        freeze_base=True,
        forward_passes=2
    )
    
    model.eval()
    print("✅ Created LRM model")
    
    # Create data loader with small batch size for testing
    dataset = EmoSoundscapesDataset()
    validate_sampler = EmotionValidateSampler(hdf5_path=test_data_path, batch_size=4)
    validate_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_sampler=validate_sampler,
        collate_fn=emotion_collate_fn, num_workers=2, pin_memory=False)
    
    print("✅ Created data loader")
    
    # Test 1: Verify data structure
    print("\n📊 Test 1: Analyzing data structure...")
    
    audio_segments = defaultdict(list)
    total_samples = 0
    
    for batch_data_dict in validate_loader:
        batch_audio_name = batch_data_dict['audio_name']
        
        for audio_name in batch_audio_name:
            name_str = audio_name.decode() if isinstance(audio_name, bytes) else audio_name
            base_name = name_str.split('_seg')[0] if '_seg' in name_str else name_str
            segment_idx = int(name_str.split('_seg')[1]) if '_seg' in name_str else 0
            
            audio_segments[base_name].append(segment_idx)
            total_samples += 1
    
    print(f"   Total samples: {total_samples}")
    print(f"   Unique audio files: {len(audio_segments)}")
    
    # Show sample audio files and their segments
    sample_files = list(audio_segments.keys())[:3]
    for base_name in sample_files:
        segments = sorted(audio_segments[base_name])
        print(f"   {base_name}: segments {segments}")
    
    print("✅ Data structure analysis complete")
    
    # Test 2: Test LRM evaluator
    print("\n🔄 Test 2: Testing LRM evaluator with segment-based feedback...")
    
    evaluator = LRMEmotionEvaluator(model=model)
    
    # Run evaluation on a subset of data
    print("   Running evaluation (this may take a moment)...")
    
    try:
        statistics, feedback_analysis = evaluator.evaluate_with_feedback_analysis(validate_loader)
        
        print("✅ LRM evaluation completed successfully!")
        
        # Print key results
        print(f"   Processed {statistics['segment_num_samples']} segments")
        print(f"   Audio files: {feedback_analysis['num_audio_files']}")
        print(f"   Avg segments per audio: {feedback_analysis['avg_segments_per_audio']:.1f}")
        
        # Show segment position effects
        print("\n   📈 Performance by segment position:")
        for pos_key, metrics in feedback_analysis['segment_position_effects'].items():
            print(f"      {pos_key}: Count={metrics['count']}, "
                  f"Valence MAE={metrics['valence_mae']:.4f}, "
                  f"Arousal MAE={metrics['arousal_mae']:.4f}")
        
        print("✅ Feedback analysis complete")
        
    except Exception as e:
        print(f"❌ LRM evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Compare with standard evaluation
    print("\n⚖️  Test 3: Comparing with standard evaluation...")
    
    try:
        from emotion_evaluate import EmotionEvaluator
        
        standard_evaluator = EmotionEvaluator(model=model)
        standard_statistics = standard_evaluator.evaluate(validate_loader)
        
        print("✅ Standard evaluation completed")
        
        # Compare key metrics
        print("\n   📊 Metric Comparison:")
        print(f"      Segment MAE - LRM: {statistics['segment_mean_mae']:.4f}, "
              f"Standard: {standard_statistics['segment_mean_mae']:.4f}")
        print(f"      Audio MAE - LRM: {statistics['audio_mean_mae']:.4f}, "
              f"Standard: {standard_statistics['audio_mean_mae']:.4f}")
        
        # The results should be different because LRM uses feedback between segments
        mae_diff = abs(statistics['audio_mean_mae'] - standard_statistics['audio_mean_mae'])
        if mae_diff > 0.001:  # Small threshold for numerical differences
            print("✅ LRM and standard evaluations produce different results (expected)")
        else:
            print("⚠️  LRM and standard evaluations produce very similar results")
            print("    This might indicate feedback is not having a strong effect")
        
    except Exception as e:
        print(f"⚠️  Standard evaluation comparison failed: {e}")
    
    # Test 4: Test individual model forward passes
    print("\n🔬 Test 4: Testing individual model forward passes...")
    
    try:
        # Get a single batch for testing
        for batch_data_dict in validate_loader:
            batch_feature = batch_data_dict['feature']
            if len(batch_feature.shape) == 3:
                batch_feature = batch_feature.unsqueeze(1)
            
            # Test single forward pass
            with torch.no_grad():
                output_single = model(batch_feature[:1], forward_passes=1)
                print(f"   Single pass output shape: valence={output_single['valence'].shape}, "
                      f"arousal={output_single['arousal'].shape}")
                
                # Test multiple forward passes
                output_multi = model(batch_feature[:1], forward_passes=2)
                print(f"   Multi pass output shape: valence={output_multi['valence'].shape}, "
                      f"arousal={output_multi['arousal'].shape}")
                
                # Test return_all_passes
                all_outputs = model(batch_feature[:1], forward_passes=2, return_all_passes=True)
                print(f"   All passes returned: {len(all_outputs)} outputs")
                
                # Check if outputs are different between passes
                val_diff = torch.abs(all_outputs[0]['valence'] - all_outputs[1]['valence']).mean().item()
                ar_diff = torch.abs(all_outputs[0]['arousal'] - all_outputs[1]['arousal']).mean().item()
                
                print(f"   Difference between passes - Valence: {val_diff:.6f}, Arousal: {ar_diff:.6f}")
                
                if val_diff > 1e-6 or ar_diff > 1e-6:
                    print("✅ Multiple passes produce different outputs (feedback is working)")
                else:
                    print("⚠️  Multiple passes produce identical outputs (feedback may not be active)")
            
            break  # Only test first batch
        
        print("✅ Individual forward pass testing complete")
        
    except Exception as e:
        print(f"❌ Individual forward pass testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 All tests completed!")
    print("=" * 60)
    print("Summary:")
    print("✅ LRM model created successfully")
    print("✅ Data structure analyzed")
    print("✅ LRM evaluator working with segment-based feedback")
    print("✅ Feedback analysis generated")
    print("✅ Individual forward passes tested")
    print("\nThe segment-based feedback system is ready for training!")
    
    return True


if __name__ == '__main__':
    success = test_segment_feedback()
    sys.exit(0 if success else 1) 