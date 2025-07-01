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

def load_real_emotion_data(hdf5_path, split='val', train_ratio=0.7, max_samples=50):
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
    
    # Limit to max_samples
    if max_samples and len(selected_indices) > max_samples:
        selected_indices = selected_indices[:max_samples]
    
    selected_features = features[selected_indices]
    selected_valence = valence[selected_indices]
    selected_arousal = arousal[selected_indices]
    selected_names = [audio_names[i] for i in selected_indices]
    
    print(f"   Selected {len(selected_indices)} {split} samples")
    
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

def debug_steering_direction():
    """Debug whether steering is moving predictions in the right direction."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, forward_passes=2
    ).to(device)
    
    # Load newly trained model
    checkpoint_path = '/home/pengliu/Private/my_panns_transfer_to_gtzan/workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print("✅ Loaded newly trained model")
    else:
        print(f"❌ Error: Model not found at {checkpoint_path}")
        return
    
    # Load steering signals
    steering_signals_path = './steering_signals_25bin/steering_signals_25bin.json'
    if not os.path.exists(steering_signals_path):
        steering_signals_path = './tmp/25bin_steering_signals/steering_signals_25bin.json'
    
    with open(steering_signals_path, 'r') as f:
        signals_25bin = json.load(f)
    
    print(f"✅ Loaded steering signals: {len(signals_25bin)} categories")
    
    # Load REAL emotion dataset (small sample for detailed analysis)
    dataset_paths = [
        'workspaces/emotion_feedback/features/emotion_features.h5',
    ]
    
    real_data = None
    for path in dataset_paths:
        if os.path.exists(path):
            real_data = load_real_emotion_data(path, split='val', max_samples=20)  # Small sample for detailed analysis
            break
    
    if real_data is None:
        print("❌ Error: Could not find real emotion dataset")
        return
    
    print(f"✅ Loaded {len(real_data)} REAL emotion validation samples")
    
    # Analyze individual samples
    print(f"\n🔍 INDIVIDUAL SAMPLE ANALYSIS")
    print(f"{'Sample':<6} {'Category':<25} {'True V':<8} {'True A':<8} {'Pred V':<8} {'Pred A':<8} {'Steer V':<8} {'Steer A':<8} {'ΔV':<8} {'ΔA':<8} {'V Dir':<6} {'A Dir':<6}")
    print("-" * 130)
    
    strength = 0.3  # Use moderate strength for detailed analysis
    correct_direction_v = 0
    correct_direction_a = 0
    total_steered = 0
    
    for sample_idx, (features, true_val, true_aro, audio_name) in enumerate(real_data):
        # Prepare input
        features = features.unsqueeze(0).to(device)
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_output = model(features, forward_passes=2, steering_signals=None, first_pass_steering=False)
            baseline_val = baseline_output['valence'][0].item()
            baseline_aro = baseline_output['arousal'][0].item()
        
        # Get TARGET CATEGORY for this specific sample
        target_category = categorize_emotion_25bin(true_val, true_aro)
        
        if target_category in signals_25bin:
            signals = signals_25bin[target_category]
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
                steered_output = model(features, 
                                     forward_passes=2,
                                     steering_signals=steering_signals_current,
                                     first_pass_steering=False)
            
            steered_val = steered_output['valence'][0].item()
            steered_aro = steered_output['arousal'][0].item()
            
            # Calculate changes
            val_change = steered_val - baseline_val
            aro_change = steered_aro - baseline_aro
            
            # Check if steering moved prediction in the right direction
            val_error_baseline = abs(true_val - baseline_val)
            val_error_steered = abs(true_val - steered_val)
            val_direction_correct = "✓" if val_error_steered < val_error_baseline else "✗"
            
            aro_error_baseline = abs(true_aro - baseline_aro)
            aro_error_steered = abs(true_aro - steered_aro)  
            aro_direction_correct = "✓" if aro_error_steered < aro_error_baseline else "✗"
            
            if val_error_steered < val_error_baseline:
                correct_direction_v += 1
            if aro_error_steered < aro_error_baseline:
                correct_direction_a += 1
            total_steered += 1
            
            print(f"{sample_idx:<6} {target_category:<25} {true_val:<8.3f} {true_aro:<8.3f} {baseline_val:<8.3f} {baseline_aro:<8.3f} {steered_val:<8.3f} {steered_aro:<8.3f} {val_change:<+8.3f} {aro_change:<+8.3f} {val_direction_correct:<6} {aro_direction_correct:<6}")
        
        else:
            print(f"{sample_idx:<6} {target_category:<25} {true_val:<8.3f} {true_aro:<8.3f} {baseline_val:<8.3f} {baseline_aro:<8.3f} {'NO_STEER':<8} {'NO_STEER':<8} {'NO_STEER':<8} {'NO_STEER':<8} {'N/A':<6} {'N/A':<6}")
    
    # Summary
    if total_steered > 0:
        v_correct_pct = (correct_direction_v / total_steered) * 100
        a_correct_pct = (correct_direction_a / total_steered) * 100
        
        print(f"\n📊 STEERING DIRECTION ANALYSIS:")
        print(f"   Samples with steering: {total_steered}")
        print(f"   Valence correct direction: {correct_direction_v}/{total_steered} ({v_correct_pct:.1f}%)")
        print(f"   Arousal correct direction: {correct_direction_a}/{total_steered} ({a_correct_pct:.1f}%)")
        
        if v_correct_pct < 50:
            print(f"   ⚠️  PROBLEM: Valence steering mostly moves predictions in WRONG direction!")
        if a_correct_pct < 50:
            print(f"   ⚠️  PROBLEM: Arousal steering mostly moves predictions in WRONG direction!")
        
        if v_correct_pct > 60 and a_correct_pct > 60:
            print(f"   ✅ Steering generally moves predictions in correct direction")
        elif v_correct_pct > 40 and a_correct_pct > 40:
            print(f"   ⚠️  Mixed results: Steering sometimes helps, sometimes hurts")
        else:
            print(f"   ❌ Major problem: Steering signals appear to be counterproductive")

if __name__ == "__main__":
    debug_steering_direction() 