#!/usr/bin/env python3

import sys
import os
# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import numpy as np
import h5py
from src.models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def test_baseline_predictions():
    """Test baseline model predictions without any steering."""
    
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
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            print(f"Using checkpoint: {path}")
            break
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print("✅ Loaded newly trained model")
    else:
        print("❌ Error: No checkpoint found")
        return
    
    # Load dataset
    dataset_paths = [
        'workspaces/emotion_feedback/features/emotion_features.h5',
    ]
    
    dataset_path = None
    for path in dataset_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"Found dataset at: {path}")
            break
    
    if not dataset_path:
        print("❌ Error: No dataset found")
        return
    
    # Load validation data
    with h5py.File(dataset_path, 'r') as hf:
        features = hf['feature'][:]
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        audio_names = [name.decode() if isinstance(name, bytes) else name for name in hf['audio_name'][:]]
    
    # Create validation split (same as training)
    np.random.seed(42)
    total_samples = len(features)
    train_size = int(total_samples * 0.7)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    val_indices = indices[train_size:]
    
    # Limit to first 20 samples for detailed analysis
    val_indices = val_indices[:20]
    
    print(f"\n🔍 Testing {len(val_indices)} validation samples (detailed analysis)")
    print(f"{'Sample':<5} {'True Val':<8} {'True Aro':<8} {'Pred Val':<8} {'Pred Aro':<8} {'Val Err':<8} {'Aro Err':<8}")
    print("-" * 70)
    
    all_pred_vals = []
    all_pred_aros = []
    all_true_vals = []
    all_true_aros = []
    
    for i, idx in enumerate(val_indices):
        # Prepare input
        feature = torch.tensor(features[idx], dtype=torch.float32).unsqueeze(0).to(device)
        true_val = valence[idx]
        true_aro = arousal[idx]
        
        # Forward pass (NO STEERING)
        with torch.no_grad():
            output = model(feature, forward_passes=2, steering_signals=None)
        
        pred_val = output['valence'][0].item()
        pred_aro = output['arousal'][0].item()
        
        val_err = abs(pred_val - true_val)
        aro_err = abs(pred_aro - true_aro)
        
        print(f"{i+1:<5} {true_val:<8.3f} {true_aro:<8.3f} {pred_val:<8.3f} {pred_aro:<8.3f} {val_err:<8.3f} {aro_err:<8.3f}")
        
        all_pred_vals.append(pred_val)
        all_pred_aros.append(pred_aro)
        all_true_vals.append(true_val)
        all_true_aros.append(true_aro)
    
    # Statistics
    mean_pred_val = np.mean(all_pred_vals)
    mean_pred_aro = np.mean(all_pred_aros)
    mean_true_val = np.mean(all_true_vals)
    mean_true_aro = np.mean(all_true_aros)
    
    val_corr = np.corrcoef(all_true_vals, all_pred_vals)[0, 1]
    aro_corr = np.corrcoef(all_true_aros, all_pred_aros)[0, 1]
    
    print(f"\n📊 Statistics:")
    print(f"   Mean True Valence: {mean_true_val:.3f}")
    print(f"   Mean Pred Valence: {mean_pred_val:.3f}")
    print(f"   Mean True Arousal: {mean_true_aro:.3f}")
    print(f"   Mean Pred Arousal: {mean_pred_aro:.3f}")
    print(f"   Valence Correlation: {val_corr:.3f}")
    print(f"   Arousal Correlation: {aro_corr:.3f}")
    
    # Check prediction ranges
    pred_val_range = (min(all_pred_vals), max(all_pred_vals))
    pred_aro_range = (min(all_pred_aros), max(all_pred_aros))
    true_val_range = (min(all_true_vals), max(all_true_vals))
    true_aro_range = (min(all_true_aros), max(all_true_aros))
    
    print(f"\n📈 Prediction Ranges:")
    print(f"   True Valence: [{true_val_range[0]:.3f}, {true_val_range[1]:.3f}]")
    print(f"   Pred Valence: [{pred_val_range[0]:.3f}, {pred_val_range[1]:.3f}]")
    print(f"   True Arousal: [{true_aro_range[0]:.3f}, {true_aro_range[1]:.3f}]")
    print(f"   Pred Arousal: [{pred_aro_range[0]:.3f}, {pred_aro_range[1]:.3f}]")
    
    # Diagnosis
    if abs(mean_pred_val) < 0.1 and abs(mean_pred_aro) < 0.1:
        print(f"\n⚠️  WARNING: Predictions are very close to zero!")
        print(f"   This suggests the model might not be trained properly.")
    elif pred_val_range[1] - pred_val_range[0] < 0.2:
        print(f"\n⚠️  WARNING: Prediction range is very narrow!")
        print(f"   This suggests the model has limited dynamic range.")
    else:
        print(f"\n✅ Model predictions look reasonable!")

if __name__ == "__main__":
    test_baseline_predictions() 