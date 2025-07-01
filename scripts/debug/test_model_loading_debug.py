#!/usr/bin/env python3

import sys
import os
# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import json
import numpy as np
import h5py
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def test_model_loading():
    """Debug script to check if the newly trained model is loading correctly."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, forward_passes=2
    ).to(device)
    
    # Load newly trained model
    checkpoint_path = '/home/pengliu/Private/my_panns_transfer_to_gtzan/workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    
    print(f"🔍 Checking model checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Model checkpoint not found!")
        return
    
    # Load checkpoint and inspect
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"✅ Checkpoint loaded successfully")
    print(f"   Checkpoint keys: {list(checkpoint.keys())}")
    
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        print(f"   Model state dict keys (first 10): {list(model_state.keys())[:10]}")
        
        # Load state dict
        model.load_state_dict(model_state)
        print("✅ Model state dict loaded")
    else:
        print("❌ Error: 'model' key not found in checkpoint")
        return
    
    # Check if there are training stats
    if 'iteration' in checkpoint:
        print(f"   Training iteration: {checkpoint['iteration']}")
    if 'statistics' in checkpoint:
        stats = checkpoint['statistics']
        print(f"   Training statistics available: {list(stats.keys())}")
        
        # Check validation performance
        if 'validate' in stats and len(stats['validate']) > 0:
            last_val = stats['validate'][-1]
            print(f"   Last validation stats: {last_val}")
    
    # Set model to eval mode
    model.eval()
    print("✅ Model set to eval mode")
    
    # Test with a few real samples
    print("\n🧪 Testing model predictions on real data...")
    
    # Load real emotion dataset
    dataset_paths = [
        'workspaces/emotion_regression/features/emotion_features.h5',
        '/DATA/pliu/EmotionData/emotion_features.h5',
        './features/emotion_features.h5'
    ]
    
    real_data = None
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"📂 Loading data from: {path}")
            with h5py.File(path, 'r') as hf:
                features = hf['feature'][:5]  # Just first 5 samples
                valence = hf['valence'][:5]
                arousal = hf['arousal'][:5]
                audio_names = [name.decode() if isinstance(name, bytes) else name for name in hf['audio_name'][:5]]
            real_data = (features, valence, arousal, audio_names)
            break
    
    if real_data is None:
        print("❌ Error: Could not find real emotion dataset")
        return
    
    features, true_valence, true_arousal, audio_names = real_data
    print(f"✅ Loaded {len(features)} test samples")
    
    # Test predictions
    print("\n📊 Sample Predictions:")
    print(f"{'Sample':>6} {'True V':>8} {'True A':>8} {'Pred V':>8} {'Pred A':>8} {'V Err':>8} {'A Err':>8}")
    print("-" * 60)
    
    with torch.no_grad():
        for i in range(len(features)):
            # Prepare input
            feature_input = torch.tensor(features[i], dtype=torch.float32).unsqueeze(0).to(device)
            
            # Forward pass (no steering)
            output = model(feature_input, forward_passes=2, steering_signals=None, first_pass_steering=False)
            
            # Get predictions
            pred_val = output['valence'][0].item()
            pred_aro = output['arousal'][0].item()
            
            # Calculate errors
            val_err = pred_val - true_valence[i]
            aro_err = pred_aro - true_arousal[i]
            
            print(f"{i+1:>6} {true_valence[i]:>8.3f} {true_arousal[i]:>8.3f} {pred_val:>8.3f} {pred_aro:>8.3f} {val_err:>+8.3f} {aro_err:>+8.3f}")
    
    # Calculate overall statistics
    all_pred_vals = []
    all_pred_aros = []
    
    with torch.no_grad():
        for i in range(len(features)):
            feature_input = torch.tensor(features[i], dtype=torch.float32).unsqueeze(0).to(device)
            output = model(feature_input, forward_passes=2, steering_signals=None, first_pass_steering=False)
            all_pred_vals.append(output['valence'][0].item())
            all_pred_aros.append(output['arousal'][0].item())
    
    # Correlations
    val_corr = np.corrcoef(all_pred_vals, true_valence)[0, 1] if len(set(all_pred_vals)) > 1 else 0.0
    aro_corr = np.corrcoef(all_pred_aros, true_arousal)[0, 1] if len(set(all_pred_aros)) > 1 else 0.0
    
    print(f"\n📈 Overall Statistics (5 samples):")
    print(f"   Valence correlation: {val_corr:.3f}")
    print(f"   Arousal correlation: {aro_corr:.3f}")
    print(f"   Mean predicted valence: {np.mean(all_pred_vals):.3f}")
    print(f"   Mean predicted arousal: {np.mean(all_pred_aros):.3f}")
    print(f"   Mean true valence: {np.mean(true_valence):.3f}")
    print(f"   Mean true arousal: {np.mean(true_arousal):.3f}")
    
    # Check if predictions are reasonable
    if abs(np.mean(all_pred_vals)) > 2.0 or abs(np.mean(all_pred_aros)) > 2.0:
        print("⚠️  WARNING: Predictions seem unreasonable (too extreme)")
    elif val_corr < 0.1 and aro_corr < 0.1:
        print("⚠️  WARNING: Very low correlations - model might not be working correctly")
    else:
        print("✅ Predictions look reasonable")

if __name__ == "__main__":
    test_model_loading() 