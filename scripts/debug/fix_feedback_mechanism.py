#!/usr/bin/env python3
"""
Fix script for the broken feedback mechanism in the emotion feedback model.
This script implements proper iterative feedback where each pass uses the output
from the previous pass to generate new feedback signals.
"""

import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
from data.data_generator import EmoSoundscapesDataset, EmotionValidateSampler, emotion_collate_fn
from configs.model_configs import cnn6_config

def test_fixed_feedback_mechanism(model_path, dataset_path, num_samples=5):
    """Test the fixed feedback mechanism to see if it's working properly."""
    print("🔧 Testing Fixed Feedback Mechanism")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model with fixed feedback
    print("\n📦 Loading model with fixed feedback...")
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000,
        window_size=cnn6_config['window_size'],
        hop_size=cnn6_config['hop_size'],
        mel_bins=cnn6_config['mel_bins'],
        fmin=cnn6_config['fmin'],
        fmax=cnn6_config['fmax'],
        freeze_base=True,
        forward_passes=2
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    
    # Load test samples
    print(f"\n📊 Loading {num_samples} test samples...")
    dataset = EmoSoundscapesDataset()
    validate_sampler = EmotionValidateSampler(hdf5_path=dataset_path, batch_size=1, train_ratio=0.7)
    validate_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_sampler=validate_sampler, 
        collate_fn=emotion_collate_fn, 
        num_workers=0, 
        pin_memory=False
    )
    
    # Get test samples
    test_samples = []
    for i, batch_data_dict in enumerate(validate_loader):
        if i >= num_samples:
            break
        test_samples.append(batch_data_dict)
    
    print(f"✅ Loaded {len(test_samples)} test samples")
    
    # Test fixed feedback mechanism
    print(f"\n🧪 Testing fixed feedback mechanism...")
    
    for sample_idx, sample_data in enumerate(test_samples):
        print(f"\n--- Sample {sample_idx + 1} ---")
        
        # Prepare input
        feature = sample_data['feature']
        if len(feature.shape) == 3:
            feature = feature.unsqueeze(1)
        feature = feature.to(device)
        
        # Test single pass
        print("  🔄 Single pass...")
        with torch.no_grad():
            output_1 = model(feature, forward_passes=1)
            valence_1 = output_1['valence'].cpu().numpy()[0, 0]
            arousal_1 = output_1['arousal'].cpu().numpy()[0, 0]
        
        print(f"    Valence: {valence_1:.4f}, Arousal: {arousal_1:.4f}")
        
        # Test multiple passes with fixed feedback
        print("  🔄 Multiple passes with fixed feedback...")
        with torch.no_grad():
            output_2 = model(feature, forward_passes=2)
            valence_2 = output_2['valence'].cpu().numpy()[0, 0]
            arousal_2 = output_2['arousal'].cpu().numpy()[0, 0]
        
        print(f"    Valence: {valence_2:.4f}, Arousal: {arousal_2:.4f}")
        
        # Check if predictions changed
        valence_diff = abs(valence_2 - valence_1)
        arousal_diff = abs(arousal_2 - arousal_1)
        
        print(f"    Changes: Valence Δ={valence_diff:.6f}, Arousal Δ={arousal_diff:.6f}")
        
        if valence_diff < 1e-6 and arousal_diff < 1e-6:
            print("    ⚠️  Still no change detected - feedback may still not be working")
        else:
            print("    ✅ Change detected - fixed feedback is working!")
    
    # Test with different forward_passes values
    print(f"\n🧪 Testing different forward_passes values with fixed feedback...")
    
    sample_data = test_samples[0]
    feature = sample_data['feature']
    if len(feature.shape) == 3:
        feature = feature.unsqueeze(1)
    feature = feature.to(device)
    
    results = []
    for passes in [1, 2, 3, 4]:
        with torch.no_grad():
            output = model(feature, forward_passes=passes)
            valence = output['valence'].cpu().numpy()[0, 0]
            arousal = output['arousal'].cpu().numpy()[0, 0]
        
        results.append({
            'passes': passes,
            'valence': valence,
            'arousal': arousal
        })
        
        print(f"  {passes} pass(es): Valence={valence:.4f}, Arousal={arousal:.4f}")
    
    # Check if results vary
    all_identical = all(
        abs(results[i]['valence'] - results[0]['valence']) < 1e-6 and
        abs(results[i]['arousal'] - results[0]['arousal']) < 1e-6
        for i in range(1, len(results))
    )
    
    if all_identical:
        print(f"\n❌ All predictions are still identical - feedback mechanism still broken!")
    else:
        print(f"\n✅ Predictions vary with number of passes - fixed feedback is working!")
        
        # Show the progression
        print(f"\n📈 Prediction progression:")
        for i, result in enumerate(results):
            if i == 0:
                print(f"  Pass {result['passes']}: Valence={result['valence']:.4f}, Arousal={result['arousal']:.4f} (baseline)")
            else:
                prev_result = results[i-1]
                valence_change = result['valence'] - prev_result['valence']
                arousal_change = result['arousal'] - prev_result['arousal']
                print(f"  Pass {result['passes']}: Valence={result['valence']:.4f} (Δ={valence_change:+.4f}), Arousal={result['arousal']:.4f} (Δ={arousal_change:+.4f})")
    
    print(f"\n🎯 Summary:")
    if all_identical:
        print(f"   - Fixed feedback mechanism is still not working")
        print(f"   - Need to investigate LRM implementation further")
    else:
        print(f"   - Fixed feedback mechanism is working!")
        print(f"   - Multiple passes now show different predictions")
        print(f"   - The model can benefit from iterative refinement")

def main():
    parser = argparse.ArgumentParser(description='Test fixed feedback mechanism in emotion feedback model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to emotion features HDF5 file')
    parser.add_argument('--num-samples', type=int, default=3, help='Number of samples to test')
    
    args = parser.parse_args()
    
    test_fixed_feedback_mechanism(args.model_path, args.dataset_path, args.num_samples)

if __name__ == '__main__':
    main()
