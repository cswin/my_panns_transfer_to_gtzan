#!/usr/bin/env python3

import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import json
import numpy as np
import h5py
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def categorize_emotion_25bin(valence, arousal):
    """Categorize emotion into 25-bin system based on valence/arousal values."""
    val_thresholds = [-0.6, -0.2, 0.2, 0.6]  
    aro_thresholds = [-0.6, -0.2, 0.2, 0.6]  
    
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

def analyze_steering_signals():
    """Analyze what the steering signals actually represent."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, forward_passes=2
    ).to(device)
    
    checkpoint_path = '/home/pengliu/Private/my_panns_transfer_to_gtzan/workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print("✅ Loaded model")
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
    
    # Load emotion dataset to get baseline samples
    dataset_paths = ['workspaces/emotion_feedback/features/emotion_features.h5']
    
    baseline_features = None
    for path in dataset_paths:
        if os.path.exists(path):
            with h5py.File(path, 'r') as hf:
                # Use first 10 samples as baseline
                baseline_features = torch.tensor(hf['feature'][:10], dtype=torch.float32).to(device)
            break
    
    if baseline_features is None:
        print("❌ Error: Could not load baseline data")
        return
    
    print(f"✅ Loaded {len(baseline_features)} baseline samples")
    
    # Test what happens when we apply each steering signal to neutral samples
    print(f"\n🔍 STEERING SIGNAL EFFECT ANALYSIS")
    print(f"Testing what each steering signal does to baseline predictions...")
    print(f"{'Category':<25} {'Expected V':<10} {'Expected A':<10} {'Actual V':<10} {'Actual A':<10} {'V Match':<8} {'A Match':<8}")
    print("-" * 95)
    
    # Get baseline predictions (no steering)
    with torch.no_grad():
        baseline_outputs = []
        for i in range(len(baseline_features)):
            output = model(baseline_features[i:i+1], forward_passes=2, steering_signals=None)
            baseline_outputs.append((output['valence'][0].item(), output['arousal'][0].item()))
    
    baseline_mean_v = np.mean([out[0] for out in baseline_outputs])
    baseline_mean_a = np.mean([out[1] for out in baseline_outputs])
    print(f"Baseline mean: V={baseline_mean_v:.3f}, A={baseline_mean_a:.3f}")
    
    # Expected emotion values for each category
    expected_emotions = {}
    for category in signals_25bin.keys():
        parts = category.split('_')
        val_part = '_'.join(parts[:-1])  # everything except last part
        aro_part = parts[-1]  # last part
        
        # Map to expected values
        val_map = {
            'very_negative': -0.8,
            'negative': -0.4,
            'neutral': 0.0,
            'positive': 0.4,
            'very_positive': 0.8
        }
        
        aro_map = {
            'very_weak': -0.8,
            'weak': -0.4,
            'moderate': 0.0,
            'strong': 0.4,
            'very_strong': 0.8
        }
        
        expected_v = val_map.get(val_part, 0.0)
        expected_a = aro_map.get(aro_part, 0.0)
        expected_emotions[category] = (expected_v, expected_a)
    
    # Test each steering signal
    for category, signals in signals_25bin.items():
        if 'valence_128d' in signals and 'arousal_128d' in signals:
            expected_v, expected_a = expected_emotions[category]
            
            # Apply steering to baseline samples
            steering_signals_current = [
                {
                    'source': 'affective_valence_128d',
                    'activation': torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device),
                    'strength': 1.0,
                    'alpha': 1.0
                },
                {
                    'source': 'affective_arousal_128d',
                    'activation': torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device),
                    'strength': 1.0,
                    'alpha': 1.0
                }
            ]
            
            steered_outputs = []
            with torch.no_grad():
                for i in range(len(baseline_features)):
                    output = model(baseline_features[i:i+1], 
                                 forward_passes=2, 
                                 steering_signals=steering_signals_current)
                    steered_outputs.append((output['valence'][0].item(), output['arousal'][0].item()))
            
            steered_mean_v = np.mean([out[0] for out in steered_outputs])
            steered_mean_a = np.mean([out[1] for out in steered_outputs])
            
            # Check if steering moves in expected direction
            v_direction_correct = "✓" if (steered_mean_v - baseline_mean_v) * (expected_v - baseline_mean_v) > 0 else "✗"
            a_direction_correct = "✓" if (steered_mean_a - baseline_mean_a) * (expected_a - baseline_mean_a) > 0 else "✗"
            
            print(f"{category:<25} {expected_v:<10.3f} {expected_a:<10.3f} {steered_mean_v:<10.3f} {steered_mean_a:<10.3f} {v_direction_correct:<8} {a_direction_correct:<8}")
    
    print(f"\n🔍 STEERING SIGNAL STATISTICS")
    print(f"Analyzing the raw steering signal values...")
    print(f"{'Category':<25} {'V Signal':<15} {'A Signal':<15} {'V Range':<15} {'A Range':<15}")
    print("-" * 90)
    
    for category, signals in signals_25bin.items():
        if 'valence_128d' in signals and 'arousal_128d' in signals:
            v_signal = np.array(signals['valence_128d'])
            a_signal = np.array(signals['arousal_128d'])
            
            v_stats = f"μ={v_signal.mean():.3f},σ={v_signal.std():.3f}"
            a_stats = f"μ={a_signal.mean():.3f},σ={a_signal.std():.3f}"
            v_range = f"[{v_signal.min():.3f},{v_signal.max():.3f}]"
            a_range = f"[{a_signal.min():.3f},{a_signal.max():.3f}]"
            
            print(f"{category:<25} {v_stats:<15} {a_stats:<15} {v_range:<15} {a_range:<15}")
    
    print(f"\n📊 RECOMMENDATIONS:")
    print(f"1. If most signals move predictions in wrong direction:")
    print(f"   → Try inverting steering signals (multiply by -1)")
    print(f"2. If signal statistics are inconsistent:")
    print(f"   → Regenerate signals with better methodology")
    print(f"3. If only some categories work:")
    print(f"   → Use only well-performing categories for steering")

if __name__ == "__main__":
    analyze_steering_signals() 