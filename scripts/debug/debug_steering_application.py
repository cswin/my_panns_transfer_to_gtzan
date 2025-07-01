#!/usr/bin/env python3
"""
Debug script to test steering signal application
"""

import sys
import os
# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import json
import h5py
import numpy as np
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def debug_steering_application():
    """Debug whether steering signals are actually being applied."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, forward_passes=2
    ).to(device)
    
    # Load newly trained model
    checkpoint_paths = [
        'workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth',
        '/home/pengliu/Private/my_panns_transfer_to_gtzan/workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            print(f"Using checkpoint: {path}")
            break
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print("✅ Loaded newly trained model")
    else:
        print(f"❌ Error: Model not found")
        return
    
    # Load steering signals
    steering_signals_path = './steering_signals_25bin/steering_signals_25bin.json'
    if not os.path.exists(steering_signals_path):
        steering_signals_path = './tmp/25bin_steering_signals/steering_signals_25bin.json'
    
    with open(steering_signals_path, 'r') as f:
        signals_25bin = json.load(f)
    
    print(f"✅ Loaded steering signals: {len(signals_25bin)} categories")
    
    # Load a test sample
    dataset_paths = [
        'workspaces/emotion_regression/features/emotion_features.h5',
        '/DATA/pliu/EmotionData/emotion_features.h5',
        './features/emotion_features.h5'
    ]
    
    test_feature = None
    for path in dataset_paths:
        if os.path.exists(path):
            with h5py.File(path, 'r') as hf:
                test_feature = torch.tensor(hf['feature'][0], dtype=torch.float32).unsqueeze(0).to(device)
            break
    
    if test_feature is None:
        print("❌ Error: Could not load test data")
        return
    
    print(f"✅ Loaded test feature: {test_feature.shape}")
    
    # Test 1: Baseline prediction (no steering)
    print("\n🧪 Test 1: Baseline (no steering)")
    with torch.no_grad():
        baseline_output = model(test_feature, forward_passes=2, steering_signals=None, first_pass_steering=False)
        baseline_val = baseline_output['valence'][0].item()
        baseline_aro = baseline_output['arousal'][0].item()
        print(f"   Baseline: Valence={baseline_val:.4f}, Arousal={baseline_aro:.4f}")
    
    # Test 2: Steering with different strengths
    category = 'very_negative_strong'
    if category not in signals_25bin:
        print(f"❌ Error: Category {category} not in steering signals")
        return
    
    signals = signals_25bin[category]
    print(f"\n🧪 Test 2: Steering with category '{category}'")
    
    strengths = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    
    for strength in strengths:
        # Prepare steering signals
        steering_signals_current = []
        
        if 'valence_128d' in signals:
            valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
            steering_signals_current.append({
                'source': 'affective_valence_128d',
                'activation': valence_signal,
                'strength': strength,
                'alpha': 1.0
            })
        
        if 'arousal_128d' in signals:
            arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
            steering_signals_current.append({
                'source': 'affective_arousal_128d',
                'activation': arousal_signal,
                'strength': strength,
                'alpha': 1.0
            })
        
        # Forward pass with steering
        with torch.no_grad():
            steered_output = model(test_feature, 
                                 forward_passes=2,
                                 steering_signals=steering_signals_current,
                                 first_pass_steering=False)
            
            steered_val = steered_output['valence'][0].item()
            steered_aro = steered_output['arousal'][0].item()
            
            val_change = steered_val - baseline_val
            aro_change = steered_aro - baseline_aro
            
            print(f"   Strength {strength:5.1f}: Val={steered_val:.4f} (Δ={val_change:+.4f}), Aro={steered_aro:.4f} (Δ={aro_change:+.4f})")
    
    # Test 3: Check LRM structure
    print(f"\n🔍 Test 3: LRM Structure Analysis")
    print(f"   Model has LRM: {hasattr(model, 'lrm')}")
    
    if hasattr(model, 'lrm'):
        print(f"   LRM type: {type(model.lrm)}")
        
        # Check LRM modules
        lrm_modules = list(model.lrm.named_children())
        print(f"   LRM modules: {len(lrm_modules)}")
        for name, module in lrm_modules:
            print(f"     - {name}: {type(module)}")
    
    # Test 4: Check if steering signals are being stored
    print(f"\n🔍 Test 4: Steering Storage Test")
    
    # Try manual steering signal injection
    if hasattr(model, 'add_steering_signal'):
        try:
            valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
            model.add_steering_signal('affective_valence_128d', valence_signal, strength=10.0, alpha=1.0)
            print("   ✅ Manual steering signal injection successful")
            
            # Test prediction after manual injection
            with torch.no_grad():
                manual_output = model(test_feature, forward_passes=2, steering_signals=None, first_pass_steering=False)
                manual_val = manual_output['valence'][0].item()
                manual_aro = manual_output['arousal'][0].item()
                
                manual_val_change = manual_val - baseline_val
                manual_aro_change = manual_aro - baseline_aro
                
                print(f"   Manual injection: Val={manual_val:.4f} (Δ={manual_val_change:+.4f}), Aro={manual_aro:.4f} (Δ={manual_aro_change:+.4f})")
        except Exception as e:
            print(f"   ❌ Manual steering injection failed: {e}")
    else:
        print("   ❌ Model doesn't have add_steering_signal method")
    
    # Test 5: Check forward pass parameters
    print(f"\n🔍 Test 5: Forward Pass Parameter Check")
    
    # Check what parameters the forward method accepts
    import inspect
    forward_sig = inspect.signature(model.forward)
    print(f"   Forward method parameters: {list(forward_sig.parameters.keys())}")
    
    # Summary
    print(f"\n📊 Summary:")
    if all(abs(val_change) < 0.001 and abs(aro_change) < 0.001 for strength in strengths):
        print("   ❌ STEERING NOT WORKING: No changes detected across different strengths")
        print("   Possible issues:")
        print("     - Steering signals not being applied in forward pass")
        print("     - LRM modules not connected properly")
        print("     - Interface mismatch between steering format and model expectation")
    else:
        print("   ✅ STEERING WORKING: Changes detected")

if __name__ == "__main__":
    debug_steering_application()
