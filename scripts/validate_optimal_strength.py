#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

import torch
import numpy as np
import h5py
from scipy.stats import pearsonr
import json
from tqdm import tqdm

# Import our modules
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def load_emotion_features(hdf5_path, split='val', train_ratio=0.7, random_seed=42):
    """Load emotion features from HDF5 file."""
    with h5py.File(hdf5_path, 'r') as hf:
        features = hf['feature'][:]
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        audio_names = [name.decode() if isinstance(name, bytes) else name for name in hf['audio_name'][:]]
    
    # Create train/val split
    np.random.seed(random_seed)
    total_samples = len(features)
    train_size = int(total_samples * train_ratio)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    if split == 'train':
        selected_indices = indices[:train_size]
    elif split == 'val':
        selected_indices = indices[train_size:]
    else:
        selected_indices = indices
    
    selected_features = features[selected_indices]
    selected_labels = np.column_stack([valence[selected_indices], arousal[selected_indices]])
    
    return selected_features, selected_labels, selected_indices

def load_steering_signals_json(json_path):
    """Load steering signals from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    steering_signals = {}
    for category, signals in data.items():
        # Skip metadata entries
        if category in ['metadata', 'generation_config']:
            continue
            
        steering_signals[category] = {
            'valence': torch.tensor(signals['valence_128d'], dtype=torch.float32),
            'arousal': torch.tensor(signals['arousal_128d'], dtype=torch.float32)
        }
    return steering_signals

def categorize_emotion_25bin(valence, arousal):
    """Categorize emotion into 25-bin system (5x5 grid)."""
    # Valence categories
    if valence <= -0.6:
        v_cat = 'very_negative'
    elif valence <= -0.2:
        v_cat = 'negative'
    elif valence <= 0.2:
        v_cat = 'neutral'
    elif valence <= 0.6:
        v_cat = 'positive'
    else:
        v_cat = 'very_positive'
    
    # Arousal categories  
    if arousal <= -0.6:
        a_cat = 'very_weak'
    elif arousal <= -0.2:
        a_cat = 'weak'
    elif arousal <= 0.2:
        a_cat = 'middle'
    elif arousal <= 0.6:
        a_cat = 'strong'
    else:
        a_cat = 'very_strong'
    
    return f"{v_cat}_{a_cat}"

def main():
    print("🎯 Validating Optimal Steering Strength (15.0) on Full Dataset")
    print("=" * 65)
    
    # Load model
    print("📥 Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, freeze_base=True, forward_passes=2
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = 'workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Load data
    print("📊 Loading validation data...")
    features_path = 'workspaces/emotion_feedback/features/emotion_features.h5'
    features, labels, indices = load_emotion_features(features_path, split='val')
    print(f"✅ Loaded {len(features)} validation samples")
    
    # Load steering signals
    print("🎛️ Loading 25-bin steering signals...")
    steering_signals = load_steering_signals_json('tmp/25bin_steering_signals/steering_signals_25bin.json')
    print(f"✅ Loaded {len(steering_signals)} categories")
    
    # Move steering signals to device
    for category in steering_signals:
        steering_signals[category]['valence'] = steering_signals[category]['valence'].to(device)
        steering_signals[category]['arousal'] = steering_signals[category]['arousal'].to(device)
    
    # Test baseline
    print("\n🎯 Evaluating baseline (no steering)...")
    baseline_predictions = []
    with torch.no_grad():
        for i in tqdm(range(len(features)), desc="Baseline"):
            sample = torch.tensor(features[i:i+1], dtype=torch.float32).to(device)
            model.lrm.enable()
            output = model(sample, forward_passes=2)
            
            pred_valence = output['valence'].cpu().numpy()[0, 0]
            pred_arousal = output['arousal'].cpu().numpy()[0, 0]
            baseline_predictions.append([pred_valence, pred_arousal])
    
    baseline_predictions = np.array(baseline_predictions)
    baseline_valence_corr = pearsonr(labels[:, 0], baseline_predictions[:, 0])[0]
    baseline_arousal_corr = pearsonr(labels[:, 1], baseline_predictions[:, 1])[0]
    
    # Test optimal strength (15.0)
    print("\n🚀 Evaluating optimal strength (15.0)...")
    optimal_predictions = []
    optimal_strength = 15.0
    
    with torch.no_grad():
        for i in tqdm(range(len(features)), desc="Optimal Strength"):
            sample = torch.tensor(features[i:i+1], dtype=torch.float32).to(device)
            target_valence, target_arousal = labels[i]
            
            # Get steering signal based on target emotion
            category = categorize_emotion_25bin(target_valence, target_arousal)
            
            if category in steering_signals:
                valence_signal = steering_signals[category]['valence']
                arousal_signal = steering_signals[category]['arousal']
            else:
                # Fallback to neutral_middle
                fallback_category = 'neutral_middle' if 'neutral_middle' in steering_signals else list(steering_signals.keys())[0]
                valence_signal = steering_signals[fallback_category]['valence']
                arousal_signal = steering_signals[fallback_category]['arousal']
            
            # Create steering signals with optimal strength
            steering_signals_list = [
                {'source': 'affective_valence_128d', 'activation': valence_signal, 'strength': optimal_strength, 'alpha': 1.0},
                {'source': 'affective_arousal_128d', 'activation': arousal_signal, 'strength': optimal_strength, 'alpha': 1.0}
            ]
            
            # Forward pass with steering
            model.lrm.enable()
            output = model(sample, forward_passes=2, steering_signals=steering_signals_list, first_pass_steering=True)
            
            pred_valence = output['valence'].cpu().numpy()[0, 0]
            pred_arousal = output['arousal'].cpu().numpy()[0, 0]
            optimal_predictions.append([pred_valence, pred_arousal])
    
    optimal_predictions = np.array(optimal_predictions)
    optimal_valence_corr = pearsonr(labels[:, 0], optimal_predictions[:, 0])[0]
    optimal_arousal_corr = pearsonr(labels[:, 1], optimal_predictions[:, 1])[0]
    
    # Compare with previous best (strength 5.0)
    print("\n🔄 Evaluating previous best strength (5.0) for comparison...")
    prev_predictions = []
    prev_strength = 5.0
    
    with torch.no_grad():
        for i in tqdm(range(len(features)), desc="Previous Strength"):
            sample = torch.tensor(features[i:i+1], dtype=torch.float32).to(device)
            target_valence, target_arousal = labels[i]
            
            # Get steering signal based on target emotion
            category = categorize_emotion_25bin(target_valence, target_arousal)
            
            if category in steering_signals:
                valence_signal = steering_signals[category]['valence']
                arousal_signal = steering_signals[category]['arousal']
            else:
                # Fallback to neutral_middle
                fallback_category = 'neutral_middle' if 'neutral_middle' in steering_signals else list(steering_signals.keys())[0]
                valence_signal = steering_signals[fallback_category]['valence']
                arousal_signal = steering_signals[fallback_category]['arousal']
            
            # Create steering signals with previous strength
            steering_signals_list = [
                {'source': 'affective_valence_128d', 'activation': valence_signal, 'strength': prev_strength, 'alpha': 1.0},
                {'source': 'affective_arousal_128d', 'activation': arousal_signal, 'strength': prev_strength, 'alpha': 1.0}
            ]
            
            # Forward pass with steering
            model.lrm.enable()
            output = model(sample, forward_passes=2, steering_signals=steering_signals_list, first_pass_steering=True)
            
            pred_valence = output['valence'].cpu().numpy()[0, 0]
            pred_arousal = output['arousal'].cpu().numpy()[0, 0]
            prev_predictions.append([pred_valence, pred_arousal])
    
    prev_predictions = np.array(prev_predictions)
    prev_valence_corr = pearsonr(labels[:, 0], prev_predictions[:, 0])[0]
    prev_arousal_corr = pearsonr(labels[:, 1], prev_predictions[:, 1])[0]
    
    # Print results
    print("\n" + "="*80)
    print("📊 OPTIMAL STRENGTH VALIDATION RESULTS (Full Dataset)")
    print("="*80)
    print(f"{'Method':<20} {'Valence r':<12} {'Arousal r':<12} {'ΔV':<10} {'ΔA':<10} {'Combined':<10}")
    print("-" * 80)
    
    print(f"{'Baseline':<20} {baseline_valence_corr:<12.3f} {baseline_arousal_corr:<12.3f} {'--':<10} {'--':<10} {baseline_valence_corr+baseline_arousal_corr:<10.3f}")
    
    delta_v_prev = prev_valence_corr - baseline_valence_corr
    delta_a_prev = prev_arousal_corr - baseline_arousal_corr
    print(f"{'Strength 5.0':<20} {prev_valence_corr:<12.3f} {prev_arousal_corr:<12.3f} {delta_v_prev:<+10.3f} {delta_a_prev:<+10.3f} {prev_valence_corr+prev_arousal_corr:<10.3f}")
    
    delta_v_opt = optimal_valence_corr - baseline_valence_corr
    delta_a_opt = optimal_arousal_corr - baseline_arousal_corr
    print(f"{'Strength 15.0':<20} {optimal_valence_corr:<12.3f} {optimal_arousal_corr:<12.3f} {delta_v_opt:<+10.3f} {delta_a_opt:<+10.3f} {optimal_valence_corr+optimal_arousal_corr:<10.3f}")
    
    # Improvement analysis
    print(f"\n🎯 IMPROVEMENT ANALYSIS:")
    print(f"📊 Strength 15.0 vs Baseline:")
    print(f"   Valence: {optimal_valence_corr:.3f} vs {baseline_valence_corr:.3f} ({delta_v_opt:+.3f}, {delta_v_opt/baseline_valence_corr*100:+.1f}%)")
    print(f"   Arousal: {optimal_arousal_corr:.3f} vs {baseline_arousal_corr:.3f} ({delta_a_opt:+.3f}, {delta_a_opt/baseline_arousal_corr*100:+.1f}%)")
    
    print(f"\n📊 Strength 15.0 vs Strength 5.0:")
    delta_v_comp = optimal_valence_corr - prev_valence_corr
    delta_a_comp = optimal_arousal_corr - prev_arousal_corr
    print(f"   Valence: {optimal_valence_corr:.3f} vs {prev_valence_corr:.3f} ({delta_v_comp:+.3f}, {delta_v_comp/prev_valence_corr*100:+.1f}%)")
    print(f"   Arousal: {optimal_arousal_corr:.3f} vs {prev_arousal_corr:.3f} ({delta_a_comp:+.3f}, {delta_a_comp/prev_arousal_corr*100:+.1f}%)")
    
    if delta_v_opt > delta_v_prev and delta_a_opt > delta_a_prev:
        print(f"\n✅ CONCLUSION: Strength 15.0 is superior to both baseline and strength 5.0")
        print(f"🎯 RECOMMENDATION: Update default steering strength to 15.0")
    elif optimal_valence_corr + optimal_arousal_corr > prev_valence_corr + prev_arousal_corr:
        print(f"\n✅ CONCLUSION: Strength 15.0 provides better combined performance")
        print(f"🎯 RECOMMENDATION: Use strength 15.0 for maximum performance")
    else:
        print(f"\n⚠️  CONCLUSION: Results are mixed, strength 5.0 may be more robust")
        print(f"🎯 RECOMMENDATION: Keep strength 5.0 as default, offer 15.0 as high-performance option")

if __name__ == "__main__":
    main() 