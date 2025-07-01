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

def categorize_emotion_9bin(valence, arousal):
    """Categorize emotion into 9-bin system (3x3 grid)."""
    # Valence categories
    if valence <= -0.33:
        v_cat = 'negative'
    elif valence <= 0.33:
        v_cat = 'neutral'
    else:
        v_cat = 'positive'
    
    # Arousal categories
    if arousal <= -0.33:
        a_cat = 'weak'
    elif arousal <= 0.33:
        a_cat = 'middle'
    else:
        a_cat = 'strong'
    
    return f"{v_cat}_{a_cat}"

def get_interpolated_steering_signal(target_valence, target_arousal, steering_signals_25bin):
    """Get interpolated steering signal for 25-bin system."""
    # Define grid points
    valence_points = [-0.8, -0.4, 0.0, 0.4, 0.8]  # very_negative, negative, neutral, positive, very_positive
    arousal_points = [-0.8, -0.4, 0.0, 0.4, 0.8]  # very_weak, weak, middle, strong, very_strong
    valence_labels = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
    arousal_labels = ['very_weak', 'weak', 'middle', 'strong', 'very_strong']
    
    # Find interpolation weights
    weights = {}
    total_weight = 0
    
    for i, v_point in enumerate(valence_points):
        for j, a_point in enumerate(arousal_points):
            # Calculate distance
            distance = np.sqrt((target_valence - v_point)**2 + (target_arousal - a_point)**2)
            if distance < 1e-6:  # Very close to grid point
                weight = 1.0
            else:
                weight = 1.0 / (distance + 0.1)  # Add small epsilon to avoid division by zero
            
            category = f"{valence_labels[i]}_{arousal_labels[j]}"
            if category in steering_signals_25bin:
                weights[category] = weight
                total_weight += weight
    
    # Normalize weights
    if total_weight > 0:
        for category in weights:
            weights[category] /= total_weight
    
    # Create interpolated signal
    interpolated_valence = torch.zeros(128)
    interpolated_arousal = torch.zeros(128)
    
    for category, weight in weights.items():
        if category in steering_signals_25bin:
            interpolated_valence += weight * steering_signals_25bin[category]['valence']
            interpolated_arousal += weight * steering_signals_25bin[category]['arousal']
    
    return interpolated_valence, interpolated_arousal

def evaluate_steering_method(model, features, labels, steering_signals, method_name, use_interpolation=False):
    """Evaluate a steering method."""
    print(f"\n🔄 Evaluating {method_name}...")
    
    predictions = []
    device = next(model.parameters()).device
    
    # Move steering signals to device
    for category in steering_signals:
        steering_signals[category]['valence'] = steering_signals[category]['valence'].to(device)
        steering_signals[category]['arousal'] = steering_signals[category]['arousal'].to(device)
    
    with torch.no_grad():
        for i in tqdm(range(len(features)), desc=f"Processing {method_name}"):
            sample = torch.tensor(features[i:i+1], dtype=torch.float32).to(device)
            target_valence, target_arousal = labels[i]
            
            # Get steering signal based on target emotion
            if use_interpolation and '25bin' in method_name:
                # Use interpolated steering for 25-bin
                valence_signal, arousal_signal = get_interpolated_steering_signal(
                    target_valence, target_arousal, steering_signals)
                valence_signal = valence_signal.to(device)
                arousal_signal = arousal_signal.to(device)
            else:
                # Use categorical steering
                if '25bin' in method_name:
                    category = categorize_emotion_25bin(target_valence, target_arousal)
                else:
                    category = categorize_emotion_9bin(target_valence, target_arousal)
                
                if category in steering_signals:
                    valence_signal = steering_signals[category]['valence']
                    arousal_signal = steering_signals[category]['arousal']
                else:
                    # Fallback to nearest category (this shouldn't happen with good coverage)
                    print(f"Warning: Category {category} not found, using neutral_middle")
                    fallback_category = 'neutral_middle' if 'neutral_middle' in steering_signals else list(steering_signals.keys())[0]
                    valence_signal = steering_signals[fallback_category]['valence']
                    arousal_signal = steering_signals[fallback_category]['arousal']
            
            # Create steering signals for both valence and arousal pathways
            steering_signals_list = [
                {'source': 'affective_valence_128d', 'activation': valence_signal, 'strength': 5.0, 'alpha': 1.0},
                {'source': 'affective_arousal_128d', 'activation': arousal_signal, 'strength': 5.0, 'alpha': 1.0}
            ]
            
            # Forward pass with steering
            model.lrm.enable()
            output = model(sample, forward_passes=2, steering_signals=steering_signals_list, first_pass_steering=True)
            
            pred_valence = output['valence'].cpu().numpy()[0, 0]
            pred_arousal = output['arousal'].cpu().numpy()[0, 0]
            predictions.append([pred_valence, pred_arousal])
    
    predictions = np.array(predictions)
    
    # Calculate correlations
    valence_corr = pearsonr(labels[:, 0], predictions[:, 0])[0]
    arousal_corr = pearsonr(labels[:, 1], predictions[:, 1])[0]
    
    return valence_corr, arousal_corr, predictions

def main():
    print("🎯 25-Bin Steering Signals Comprehensive Evaluation (Fixed)")
    print("=" * 60)
    
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
    print("🎛️ Loading steering signals...")
    steering_9bin = load_steering_signals_json('tmp/steering_signals_by_category.json')
    steering_25bin = load_steering_signals_json('tmp/25bin_steering_signals/steering_signals_25bin.json')
    
    print(f"✅ Loaded 9-bin steering signals: {len(steering_9bin)} categories")
    print(f"✅ Loaded 25-bin steering signals: {len(steering_25bin)} categories")
    
    # Baseline evaluation (no steering)
    print("\n🔄 Evaluating baseline (no steering)...")
    baseline_predictions = []
    with torch.no_grad():
        for i in tqdm(range(len(features)), desc="Processing baseline"):
            sample = torch.tensor(features[i:i+1], dtype=torch.float32).to(device)
            model.lrm.enable()
            output = model(sample, forward_passes=2)
            
            pred_valence = output['valence'].cpu().numpy()[0, 0]
            pred_arousal = output['arousal'].cpu().numpy()[0, 0]
            baseline_predictions.append([pred_valence, pred_arousal])
    
    baseline_predictions = np.array(baseline_predictions)
    baseline_valence_corr = pearsonr(labels[:, 0], baseline_predictions[:, 0])[0]
    baseline_arousal_corr = pearsonr(labels[:, 1], baseline_predictions[:, 1])[0]
    
    # Evaluate steering methods
    results = {}
    
    # 9-bin categorical
    valence_corr, arousal_corr, _ = evaluate_steering_method(
        model, features, labels, steering_9bin, "9-bin categorical", use_interpolation=False)
    results['9-bin categorical'] = (valence_corr, arousal_corr)
    
    # 25-bin categorical  
    valence_corr, arousal_corr, _ = evaluate_steering_method(
        model, features, labels, steering_25bin, "25-bin categorical", use_interpolation=False)
    results['25-bin categorical'] = (valence_corr, arousal_corr)
    
    # 25-bin interpolation
    valence_corr, arousal_corr, _ = evaluate_steering_method(
        model, features, labels, steering_25bin, "25-bin interpolation", use_interpolation=True)
    results['25-bin interpolation'] = (valence_corr, arousal_corr)
    
    # Print results
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE STEERING EVALUATION RESULTS")
    print("="*80)
    print(f"{'Method':<20} {'Valence r':<12} {'Arousal r':<12} {'ΔV':<10} {'ΔA':<10}")
    print("-" * 80)
    
    print(f"{'Baseline':<20} {baseline_valence_corr:<12.3f} {baseline_arousal_corr:<12.3f} {'--':<10} {'--':<10}")
    
    for method, (val_corr, ar_corr) in results.items():
        delta_v = val_corr - baseline_valence_corr
        delta_a = ar_corr - baseline_arousal_corr
        print(f"{method:<20} {val_corr:<12.3f} {ar_corr:<12.3f} {delta_v:<+10.3f} {delta_a:<+10.3f}")
    
    print("\n🎯 ANALYSIS:")
    print("- Positive ΔV/ΔA means steering improved correlation")
    print("- 25-bin methods should show more fine-grained control")
    print("- Interpolation should provide smoother transitions between categories")
    
    # Find best method
    best_method = max(results.keys(), key=lambda x: sum(results[x]))
    best_valence, best_arousal = results[best_method]
    print(f"\n🏆 Best method: {best_method}")
    print(f"   Valence r: {best_valence:.3f} (Δ: {best_valence - baseline_valence_corr:+.3f})")
    print(f"   Arousal r: {best_arousal:.3f} (Δ: {best_arousal - baseline_arousal_corr:+.3f})")

if __name__ == "__main__":
    main() 