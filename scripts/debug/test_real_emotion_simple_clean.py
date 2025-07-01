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
from src.models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
from src.data.data_generator import EmoSoundscapesDataset, EmotionValidateSampler, emotion_collate_fn

def categorize_emotion_25bin(valence, arousal):
    """Categorize emotion into 25-bin system based on valence/arousal values."""
    
    # Define thresholds for 5x5 grid
    val_thresholds = [-0.6, -0.2, 0.2, 0.6]  # Creates 5 bins
    aro_thresholds = [-0.6, -0.2, 0.2, 0.6]  # Creates 5 bins
    
    # Determine valence category
    if valence <= val_thresholds[0]:
        val_cat = "very_negative"
    elif valence <= val_thresholds[1]:
        val_cat = "negative"
    elif valence <= val_thresholds[2]:
        val_cat = "neutral"
    elif valence <= val_thresholds[3]:
        val_cat = "positive"
    else:
        val_cat = "very_positive"
    
    # Determine arousal category
    if arousal <= aro_thresholds[0]:
        aro_cat = "very_weak"
    elif arousal <= aro_thresholds[1]:
        aro_cat = "weak"
    elif arousal <= aro_thresholds[2]:
        aro_cat = "moderate"
    elif arousal <= aro_thresholds[3]:
        aro_cat = "strong"
    else:
        aro_cat = "very_strong"
    
    return f"{val_cat}_{aro_cat}"

def load_real_emotion_data(hdf5_path, split='val', train_ratio=0.7, max_samples=None):
    """Load real emotion features from HDF5 file."""
    print(f"🔍 Loading real emotion data from: {hdf5_path}")
    
    if not os.path.exists(hdf5_path):
        print(f"❌ Error: Dataset file not found: {hdf5_path}")
        return None
    
    with h5py.File(hdf5_path, 'r') as hf:
        features = hf['feature'][:]
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        audio_names = [name.decode() if isinstance(name, bytes) else name for name in hf['audio_name'][:]]
        
        print(f"   Total samples in dataset: {len(features)}")
        print(f"   Feature shape: {features.shape}")
        print(f"   Valence range: [{np.min(valence):.3f}, {np.max(valence):.3f}]")
        print(f"   Arousal range: [{np.min(arousal):.3f}, {np.max(arousal):.3f}]")
    
    # Create train/val split (same as used in training)
    np.random.seed(42)  # Fixed seed for reproducible split
    total_samples = len(features)
    train_size = int(total_samples * train_ratio)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    if split == 'val':
        selected_indices = indices[train_size:]
    else:
        selected_indices = indices[:train_size]
    
    # Limit to max_samples for quick testing (None = use all samples)
    if max_samples and len(selected_indices) > max_samples:
        selected_indices = selected_indices[:max_samples]
    
    selected_features = features[selected_indices]
    selected_valence = valence[selected_indices]
    selected_arousal = arousal[selected_indices]
    selected_names = [audio_names[i] for i in selected_indices]
    
    if max_samples:
        print(f"   Selected {len(selected_indices)} {split} samples (limited to {max_samples})")
    else:
        print(f"   Selected {len(selected_indices)} {split} samples (all available)")
    
    # Convert to list of tuples for easier processing
    real_data = []
    for i in range(len(selected_features)):
        real_data.append((
            torch.tensor(selected_features[i], dtype=torch.float32),
            selected_valence[i],
            selected_arousal[i],
            selected_names[i]
        ))
    
    return real_data

def test_simple_clean():
    """Simple, clean test matching Test_Steering_Emotion.py exactly - using REAL emotion data."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, forward_passes=2
    ).to(device)
    
    # Load newly trained model with corrected steering
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
    
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()  # CRITICAL: Set to eval mode
        print("✅ Loaded newly trained model with corrected steering")
    else:
        print(f"❌ Error: New model not found in any of these locations:")
        for path in checkpoint_paths:
            print(f"   - {path}")
        return
    
    # Load steering signals
    steering_signals_path = './steering_signals_25bin/steering_signals_25bin.json'
    if not os.path.exists(steering_signals_path):
        steering_signals_path = './tmp/25bin_steering_signals/steering_signals_25bin.json'
    
    with open(steering_signals_path, 'r') as f:
        signals_25bin = json.load(f)
    
    print(f"✅ Loaded steering signals: {len(signals_25bin)} categories")
    
    # Load REAL emotion dataset
    dataset_paths = [
        'workspaces/emotion_feedback/features/emotion_features.h5',
        'workspaces/emotion_regression/features/emotion_features.h5',
        '/DATA/pliu/EmotionData/emotion_features.h5',
        './features/emotion_features.h5'
    ]
    
    real_data = None
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"Found dataset at: {path}")
            real_data = load_real_emotion_data(path, split='val', max_samples=None)  # Use all validation samples
            break
    
    if real_data is None:
        print("❌ Error: Could not find real emotion dataset in any of these locations:")
        for path in dataset_paths:
            print(f"   - {path}")
        print("Please update the dataset_paths list with the correct path to your emotion_features.h5 file")
        return
    
    print(f"✅ Loaded {len(real_data)} REAL emotion validation samples")
    
    # Test different strengths (including more fine-grained values)
    strengths = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}
    
    # Use fixed category (like Test_Steering_Emotion.py with fixed_category_steering_signal)
    fixed_category = 'very_negative_strong'
    
    for strength in strengths:
        print(f"\n🔍 Testing strength: {strength}")
        
        predictions = []
        steering_count = 0
        
        for sample_idx, (features, true_val, true_aro, audio_name) in enumerate(real_data):
            # Prepare input (like Test_Steering_Emotion.py)
            features = features.unsqueeze(0).to(device)  # [1, time_steps, mel_bins]
            
            # Prepare steering signals (EXACTLY like Test_Steering_Emotion.py)
            if strength > 0.0 and fixed_category in signals_25bin:
                signals = signals_25bin[fixed_category]
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
                
                steering_count += 1
            else:
                steering_signals_current = None
            
            # Forward pass (EXACTLY like Test_Steering_Emotion.py)
            with torch.no_grad():
                output = model(features, 
                             forward_passes=2,
                             steering_signals=steering_signals_current,
                             first_pass_steering=False)  # Like args.is_priming=False
            
            # Store results
            pred_val = output['valence'][0].item()
            pred_aro = output['arousal'][0].item()
            predictions.append((pred_val, pred_aro, true_val, true_aro, audio_name))
        
        # Calculate metrics
        pred_vals = [p[0] for p in predictions]
        pred_aros = [p[1] for p in predictions]
        true_vals = [p[2] for p in predictions]
        true_aros = [p[3] for p in predictions]
        
        # Stats
        mean_pred_val = np.mean(pred_vals)
        mean_pred_aro = np.mean(pred_aros)
        mean_true_val = np.mean(true_vals)
        mean_true_aro = np.mean(true_aros)
        
        coverage = steering_count / len(real_data) * 100
        
        results[strength] = {
            'mean_pred_val': mean_pred_val,
            'mean_pred_aro': mean_pred_aro,
            'coverage': coverage
        }
        
        print(f"   Samples: {len(predictions)}, Steering coverage: {coverage:.1f}%")
        print(f"   Valence: mean_pred={mean_pred_val:.3f}, mean_true={mean_true_val:.3f}")
        print(f"   Arousal: mean_pred={mean_pred_aro:.3f}, mean_true={mean_true_aro:.3f}")
    
    # Compare results
    print(f"\n📊 Results Comparison (REAL EMOTION DATA):")
    print(f"{'Strength':>8} {'Val Pred':>10} {'Aro Pred':>10} {'Val Δ':>8} {'Aro Δ':>8}")
    print("-" * 50)
    
    baseline = results[0.0]
    for strength in strengths:
        res = results[strength]
        val_delta = res['mean_pred_val'] - baseline['mean_pred_val']
        aro_delta = res['mean_pred_aro'] - baseline['mean_pred_aro']
        
        print(f"{strength:>8.1f} {res['mean_pred_val']:>10.3f} {res['mean_pred_aro']:>10.3f} "
              f"{val_delta:>+8.3f} {aro_delta:>+8.3f}")
    
    # Check for effects
    max_val_effect = max(abs(results[s]['mean_pred_val'] - baseline['mean_pred_val']) for s in strengths[1:])
    max_aro_effect = max(abs(results[s]['mean_pred_aro'] - baseline['mean_pred_aro']) for s in strengths[1:])
    
    print(f"\n✅ Summary (REAL EMOTION DATASET):")
    print(f"   Maximum valence effect: {max_val_effect:.4f}")
    print(f"   Maximum arousal effect: {max_aro_effect:.4f}")
    
    if max_val_effect > 0.01 or max_aro_effect > 0.01:
        print("   🎉 SUCCESS: Steering effects detected!")
    else:
        print("   ❌ FAILED: No steering effects detected")
    
    # Show a few sample predictions for debugging
    print(f"\n🔍 Sample predictions (strength={strengths[-1]}):")
    sample_predictions = predictions[:5]  # First 5 samples
    for i, (pred_v, pred_a, true_v, true_a, name) in enumerate(sample_predictions):
        print(f"   Sample {i+1}: {name[:30]}...")
        print(f"      True: V={true_v:.3f}, A={true_a:.3f}")
        print(f"      Pred: V={pred_v:.3f}, A={pred_a:.3f}")

if __name__ == "__main__":
    test_simple_clean() 