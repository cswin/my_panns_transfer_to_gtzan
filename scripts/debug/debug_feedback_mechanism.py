#!/usr/bin/env python3
"""
Debug script to investigate the feedback mechanism in the emotion feedback model.
This script will help identify why multiple passes don't show performance improvements.
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

def debug_feedback_mechanism(model_path, dataset_path, num_samples=10):
    """Debug the feedback mechanism to see if it's working properly."""
    print("🔍 Debugging Feedback Mechanism")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\n📦 Loading model...")
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
    
    # Check if LRM exists
    print(f"\n🔍 Checking LRM component...")
    if hasattr(model, 'lrm'):
        print(f"✅ LRM found: {type(model.lrm)}")
        print(f"   - LRM enabled: {getattr(model.lrm, 'enabled', 'Unknown')}")
        print(f"   - Feedback enabled: {getattr(model.lrm, 'feedback_enabled', 'Unknown')}")
        
        # Check LRM parameters
        if hasattr(model.lrm, 'modulation_layers'):
            print(f"   - Modulation layers: {len(model.lrm.modulation_layers)}")
            for i, layer in enumerate(model.lrm.modulation_layers):
                print(f"     Layer {i}: {type(layer)}")
    else:
        print("❌ LRM not found in model!")
        return
    
    # Load a few samples for testing
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
    
    # Get a few samples
    test_samples = []
    for i, batch_data_dict in enumerate(validate_loader):
        if i >= num_samples:
            break
        test_samples.append(batch_data_dict)
    
    print(f"✅ Loaded {len(test_samples)} test samples")
    
    # Test feedback mechanism
    print(f"\n🧪 Testing feedback mechanism...")
    
    for sample_idx, sample_data in enumerate(test_samples):
        print(f"\n--- Sample {sample_idx + 1} ---")
        
        # Prepare input
        feature = sample_data['feature']
        if len(feature.shape) == 3:
            feature = feature.unsqueeze(1)
        feature = feature.to(device)
        
        # Clear any stored activations
        if hasattr(model, 'lrm'):
            model.lrm.clear_stored_activations()
        
        # Test single pass
        print("  🔄 Single pass...")
        with torch.no_grad():
            output_1 = model(feature, forward_passes=1)
            valence_1 = output_1['valence'].cpu().numpy()[0, 0]
            arousal_1 = output_1['arousal'].cpu().numpy()[0, 0]
        
        print(f"    Valence: {valence_1:.4f}, Arousal: {arousal_1:.4f}")
        
        # Check if activations were stored
        if hasattr(model, 'lrm') and hasattr(model.lrm, 'stored_activations'):
            stored_count = len(model.lrm.stored_activations) if model.lrm.stored_activations else 0
            print(f"    Stored activations after pass 1: {stored_count}")
        
        # Test multiple passes
        print("  🔄 Multiple passes...")
        with torch.no_grad():
            output_2 = model(feature, forward_passes=2)
            valence_2 = output_2['valence'].cpu().numpy()[0, 0]
            arousal_2 = output_2['arousal'].cpu().numpy()[0, 0]
        
        print(f"    Valence: {valence_2:.4f}, Arousal: {arousal_2:.4f}")
        
        # Check if activations were stored
        if hasattr(model, 'lrm') and hasattr(model.lrm, 'stored_activations'):
            stored_count = len(model.lrm.stored_activations) if model.lrm.stored_activations else 0
            print(f"    Stored activations after pass 2: {stored_count}")
        
        # Check if predictions changed
        valence_diff = abs(valence_2 - valence_1)
        arousal_diff = abs(arousal_2 - arousal_1)
        
        print(f"    Changes: Valence Δ={valence_diff:.6f}, Arousal Δ={arousal_diff:.6f}")
        
        if valence_diff < 1e-6 and arousal_diff < 1e-6:
            print("    ⚠️  No change detected - feedback may not be working")
        else:
            print("    ✅ Change detected - feedback is working")
    
    # Test with different forward_passes values
    print(f"\n🧪 Testing different forward_passes values...")
    
    sample_data = test_samples[0]
    feature = sample_data['feature']
    if len(feature.shape) == 3:
        feature = feature.unsqueeze(1)
    feature = feature.to(device)
    
    results = []
    for passes in [1, 2, 3, 4]:
        # Clear stored activations
        if hasattr(model, 'lrm'):
            model.lrm.clear_stored_activations()
        
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
    
    # Check if all results are identical
    all_identical = all(
        abs(results[i]['valence'] - results[0]['valence']) < 1e-6 and
        abs(results[i]['arousal'] - results[0]['arousal']) < 1e-6
        for i in range(1, len(results))
    )
    
    if all_identical:
        print(f"\n❌ All predictions are identical - feedback mechanism is not working!")
        print(f"   This explains why multiple passes don't improve performance.")
    else:
        print(f"\n✅ Predictions vary with number of passes - feedback is working!")
    
    # Additional debugging: Check model's forward_passes parameter
    print(f"\n🔍 Checking model configuration...")
    print(f"   Model forward_passes parameter: {getattr(model, 'forward_passes', 'Not found')}")
    
    # Check if the model actually uses the forward_passes parameter
    print(f"\n🔍 Checking if model uses forward_passes parameter...")
    
    # Try to inspect the forward method
    if hasattr(model, 'forward'):
        import inspect
        forward_source = inspect.getsource(model.forward)
        if 'forward_passes' in forward_source:
            print("   ✅ forward_passes parameter is used in forward method")
        else:
            print("   ❌ forward_passes parameter is NOT used in forward method")
    
    print(f"\n🎯 Summary:")
    print(f"   - If predictions are identical across passes: Feedback mechanism is broken")
    print(f"   - If predictions vary: Feedback mechanism is working")
    print(f"   - Check the model's forward method implementation")
    print(f"   - Verify that LRM feedback is properly enabled")

def main():
    parser = argparse.ArgumentParser(description='Debug feedback mechanism in emotion feedback model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to emotion features HDF5 file')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples to test')
    
    args = parser.parse_args()
    
    debug_feedback_mechanism(args.model_path, args.dataset_path, args.num_samples)

if __name__ == '__main__':
    main() 