#!/usr/bin/env python3
"""
Comprehensive 25-bin steering signals test: Categorical vs 9-bin comparison.

This script provides a thorough comparison of:
1. 9-bin categorical (baseline)
2. 25-bin categorical (finer granularity)  
3. Baseline (no steering)
"""

import os
import sys
import json
import h5py
import torch
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
import argparse

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM


def categorize_9bin(valence, arousal):
    """Original 9-bin categorization for comparison."""
    # Valence categorization (3 bins)
    if valence < -0.33:
        v_cat = "negative"
    elif valence > 0.33:
        v_cat = "positive"
    else:
        v_cat = "neutral"
    
    # Arousal categorization (3 bins)
    if arousal < -0.33:
        a_cat = "weak"
    elif arousal > 0.33:
        a_cat = "strong"
    else:
        a_cat = "middle"
    
    return f"{v_cat}_{a_cat}"


def categorize_25bin_with_fallback(valence, arousal):
    """25-bin categorization with smart fallback."""
    # Valence categorization (5 bins)
    if valence < -0.6:
        v_cat = "very_negative"
    elif valence < -0.2:
        v_cat = "negative"
    elif valence < 0.2:
        v_cat = "neutral"
    elif valence < 0.6:
        v_cat = "positive"
    else:
        v_cat = "very_positive"
    
    # Arousal categorization (5 bins)
    if arousal < -0.6:
        a_cat = "very_weak"
    elif arousal < -0.2:
        a_cat = "weak"
    elif arousal < 0.2:
        a_cat = "middle"
    elif arousal < 0.6:
        a_cat = "strong"
    else:
        a_cat = "very_strong"
    
    category = f"{v_cat}_{a_cat}"
    
    # Smart fallback for missing categories
    fallback_mapping = {
        'very_positive_very_strong': 'very_positive_strong',
        'negative_very_weak': 'neutral_very_weak'
    }
    
    if category in fallback_mapping:
        category = fallback_mapping[category]
    
    return category


def get_category_center(category):
    """Get the center coordinates of a category in valence-arousal space."""
    # Extract valence part
    if 'very_negative' in category:
        valence = -0.8
    elif 'negative' in category:
        valence = -0.4
    elif 'neutral' in category:
        valence = 0.0
    elif 'positive' in category and 'very_positive' not in category:
        valence = 0.4
    elif 'very_positive' in category:
        valence = 0.8
    else:
        valence = 0.0
    
    # Extract arousal part
    if 'very_weak' in category:
        arousal = -0.8
    elif 'weak' in category and 'very_weak' not in category:
        arousal = -0.4
    elif 'middle' in category:
        arousal = 0.0
    elif 'strong' in category and 'very_strong' not in category:
        arousal = 0.4
    elif 'very_strong' in category:
        arousal = 0.8
    else:
        arousal = 0.0
    
    return np.array([valence, arousal])


def select_interpolated_25bin(target_valence, target_arousal, steering_signals, 
                             k_neighbors=3, distance_power=2):
    """Select steering signal using interpolation of k nearest 25-bin categories."""
    target_point = np.array([target_valence, target_arousal])
    
    # Get all category centers
    categories = list(steering_signals.keys())
    if 'metadata' in categories:
        categories.remove('metadata')
    if 'generation_config' in categories:
        categories.remove('generation_config')
    
    category_centers = []
    for category in categories:
        center = get_category_center(category)
        category_centers.append(center)
    
    category_centers = np.array(category_centers)
    
    # Calculate distances to all categories
    distances = cdist([target_point], category_centers, metric='euclidean')[0]
    
    # Get k nearest neighbors
    nearest_indices = np.argsort(distances)[:k_neighbors]
    nearest_categories = [categories[i] for i in nearest_indices]
    nearest_distances = distances[nearest_indices]
    
    # Avoid division by zero
    nearest_distances = np.maximum(nearest_distances, 1e-8)
    
    # Calculate weights (inverse distance weighting)
    weights = 1.0 / (nearest_distances ** distance_power)
    weights = weights / np.sum(weights)  # Normalize
    
    # Interpolate signals
    interpolated_signals = {}
    signal_names = list(steering_signals[nearest_categories[0]].keys())
    
    for signal_name in signal_names:
        interpolated_signal = np.zeros_like(steering_signals[nearest_categories[0]][signal_name])
        
        for i, category in enumerate(nearest_categories):
            if category in steering_signals and signal_name in steering_signals[category]:
                signal = np.array(steering_signals[category][signal_name])
                interpolated_signal += weights[i] * signal
        
        interpolated_signals[signal_name] = interpolated_signal
    
    return interpolated_signals, nearest_categories, weights


def load_emotion_model(checkpoint_path, device):
    """Load the emotion model from checkpoint."""
    print(f"🔧 Loading model from {checkpoint_path}")
    
    # Model configuration (same as training)
    model_config = {
        'sample_rate': 32000,
        'window_size': 1024,
        'hop_size': 320, 
        'mel_bins': 64,
        'fmin': 50,
        'fmax': 14000,
        'freeze_base': True
    }
    
    # Create model
    model = FeatureEmotionRegression_Cnn6_LRM(**model_config)
    
    # Load checkpoint (handle PyTorch security warning)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model


def load_dataset(dataset_path, train_ratio=0.7, random_seed=42):
    """Load and split dataset."""
    print(f"📂 Loading dataset from {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as hf:
        features = hf['feature'][:]
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        audio_names = [name.decode('utf-8') for name in hf['audio_name'][:]]
    
    # Create reproducible train/validation split
    np.random.seed(random_seed)
    total_samples = len(features)
    indices = np.random.permutation(total_samples)
    
    train_size = int(total_samples * train_ratio)
    val_indices = indices[train_size:]
    
    # Return validation set for testing
    val_features = features[val_indices]
    val_valence = valence[val_indices]
    val_arousal = arousal[val_indices]
    val_names = [audio_names[i] for i in val_indices]
    
    print(f"📊 Dataset loaded: {len(val_features)} validation samples")
    return val_features, val_valence, val_arousal, val_names


def test_steering_method(model, features, valence_targets, arousal_targets, 
                        steering_signals_9bin, steering_signals_25bin, device, num_samples=30):
    """Test steering methods: baseline, 9-bin categorical, and 25-bin categorical."""
    print(f"🧪 Testing steering methods on {num_samples} samples...")
    
    # Limit samples
    if num_samples < len(features):
        indices = np.random.choice(len(features), num_samples, replace=False)
        features = features[indices]
        valence_targets = valence_targets[indices]
        arousal_targets = arousal_targets[indices]
    
    results = {
        '9bin_categorical': {'valence': [], 'arousal': [], 'success': 0},
        '25bin_categorical': {'valence': [], 'arousal': [], 'success': 0},
        'baseline': {'valence': [], 'arousal': []},
        'targets': {'valence': valence_targets.tolist(), 'arousal': arousal_targets.tolist()}
    }
    
    for i in range(len(features)):
        sample_tensor = torch.tensor(features[i:i+1], dtype=torch.float32).to(device)
        target_v = valence_targets[i]
        target_a = arousal_targets[i]
        
        # Baseline (no steering)
        with torch.no_grad():
            baseline_output = model(sample_tensor)
        results['baseline']['valence'].append(baseline_output['valence'].cpu().item())
        results['baseline']['arousal'].append(baseline_output['arousal'].cpu().item())
        
        # Method 1: 9-bin categorical
        category_9bin = categorize_9bin(target_v, target_a)
        if category_9bin in steering_signals_9bin:
            signals = steering_signals_9bin[category_9bin]
            
            # Clear previous state and apply steering signals
            model.clear_feedback_state()
            
            if 'valence_128d' in signals:
                model.add_steering_signal(
                    source='affective_valence_128d',
                    activation=torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device),
                    strength=5.0,
                    alpha=1.0
                )
            
            if 'arousal_128d' in signals:
                model.add_steering_signal(
                    source='affective_arousal_128d',
                    activation=torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device),
                    strength=5.0,
                    alpha=1.0
                )
            
            model.lrm.enable()
            
            with torch.no_grad():
                output = model(sample_tensor, forward_passes=2)
            results['9bin_categorical']['valence'].append(output['valence'].cpu().item())
            results['9bin_categorical']['arousal'].append(output['arousal'].cpu().item())
            results['9bin_categorical']['success'] += 1
        else:
            results['9bin_categorical']['valence'].append(baseline_output['valence'].cpu().item())
            results['9bin_categorical']['arousal'].append(baseline_output['arousal'].cpu().item())
        
        # Method 2: 25-bin categorical
        category_25bin = categorize_25bin_with_fallback(target_v, target_a)
        if category_25bin in steering_signals_25bin:
            signals = steering_signals_25bin[category_25bin]
            
            # Clear previous state and apply steering signals
            model.clear_feedback_state()
            
            if 'valence_128d' in signals:
                model.add_steering_signal(
                    source='affective_valence_128d',
                    activation=torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device),
                    strength=5.0,
                    alpha=1.0
                )
            
            if 'arousal_128d' in signals:
                model.add_steering_signal(
                    source='affective_arousal_128d',
                    activation=torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device),
                    strength=5.0,
                    alpha=1.0
                )
            
            model.lrm.enable()
            
            with torch.no_grad():
                output = model(sample_tensor, forward_passes=2)
            results['25bin_categorical']['valence'].append(output['valence'].cpu().item())
            results['25bin_categorical']['arousal'].append(output['arousal'].cpu().item())
            results['25bin_categorical']['success'] += 1
        else:
            results['25bin_categorical']['valence'].append(baseline_output['valence'].cpu().item())
            results['25bin_categorical']['arousal'].append(baseline_output['arousal'].cpu().item())
    
    return results


def calculate_metrics(results):
    """Calculate performance metrics for all methods."""
    metrics = {}
    targets_v = np.array(results['targets']['valence'])
    targets_a = np.array(results['targets']['arousal'])
    
    for method_name in ['baseline', '9bin_categorical', '25bin_categorical']:
        if method_name in results:
            preds_v = np.array(results[method_name]['valence'])
            preds_a = np.array(results[method_name]['arousal'])
            
            # Calculate correlations
            corr_v = np.corrcoef(preds_v, targets_v)[0, 1] if len(preds_v) > 1 else 0.0
            corr_a = np.corrcoef(preds_a, targets_a)[0, 1] if len(preds_a) > 1 else 0.0
            
            # Calculate RMSE
            rmse_v = np.sqrt(mean_squared_error(targets_v, preds_v))
            rmse_a = np.sqrt(mean_squared_error(targets_a, preds_a))
            
            # Success rate
            success_rate = results[method_name].get('success', 0) / len(targets_v) if 'success' in results[method_name] else 1.0
            
            metrics[method_name] = {
                'valence_corr': corr_v,
                'arousal_corr': corr_a,
                'valence_rmse': rmse_v,
                'arousal_rmse': rmse_a,
                'success_rate': success_rate
            }
    
    return metrics


def print_comparison(metrics):
    """Print comprehensive comparison results."""
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE STEERING METHODS COMPARISON")
    print("="*80)
    
    print(f"\n📊 **PERFORMANCE METRICS**")
    print("-" * 60)
    
    methods = ['baseline', '9bin_categorical', '25bin_categorical']
    method_names = ['Baseline (No Steering)', '9-Bin Categorical', '25-Bin Categorical']
    
    for method, name in zip(methods, method_names):
        if method in metrics:
            m = metrics[method]
            print(f"\n**{name.upper()}:**")
            print(f"  Valence:  r={m['valence_corr']:.3f}, RMSE={m['valence_rmse']:.3f}")
            print(f"  Arousal:  r={m['arousal_corr']:.3f}, RMSE={m['arousal_rmse']:.3f}")
            if 'success_rate' in m and method != 'baseline':
                print(f"  Success Rate: {m['success_rate']:.1%}")
    
    # Performance improvements over baseline
    if 'baseline' in metrics:
        baseline = metrics['baseline']
        print(f"\n🔄 **IMPROVEMENTS OVER BASELINE:**")
        print("-" * 40)
        
        for method, name in zip(methods[1:], method_names[1:]):
            if method in metrics:
                m = metrics[method]
                v_improve = m['valence_corr'] - baseline['valence_corr']
                a_improve = m['arousal_corr'] - baseline['arousal_corr']
                v_rmse_improve = baseline['valence_rmse'] - m['valence_rmse']
                a_rmse_improve = baseline['arousal_rmse'] - m['arousal_rmse']
                
                print(f"\n**{name}:**")
                print(f"  Valence: Δr={v_improve:+.4f}, ΔRMSE={v_rmse_improve:+.4f}")
                print(f"  Arousal: Δr={a_improve:+.4f}, ΔRMSE={a_rmse_improve:+.4f}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description='Comprehensive 25-bin steering signals comparison')
    parser.add_argument('--dataset_path', type=str, 
                       default='workspaces/emotion_feedback/features/emotion_features.h5',
                       help='Path to emotion features HDF5 file')
    parser.add_argument('--model_checkpoint', type=str,
                       default='workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--steering_9bin', type=str,
                       default='tmp/steering_signals_by_category.json',
                       help='Path to 9-bin steering signals JSON file')
    parser.add_argument('--steering_25bin', type=str,
                       default='tmp/25bin_steering_signals/steering_signals_25bin.json',
                       help='Path to 25-bin steering signals JSON file')
    parser.add_argument('--num_samples', type=int, default=30, help='Number of validation samples to test')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    
    args = parser.parse_args()
    
    print("🚀 COMPREHENSIVE 25-BIN STEERING COMPARISON")
    print("=" * 60)
    
    # Setup device
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Load steering signals
    print(f"📂 Loading 9-bin steering signals from {args.steering_9bin}")
    with open(args.steering_9bin, 'r') as f:
        steering_signals_9bin = json.load(f)
    
    print(f"📂 Loading 25-bin steering signals from {args.steering_25bin}")
    with open(args.steering_25bin, 'r') as f:
        steering_signals_25bin = json.load(f)
    
    categories_9bin = [k for k in steering_signals_9bin.keys() if k not in ['metadata', 'generation_config']]
    categories_25bin = [k for k in steering_signals_25bin.keys() if k not in ['metadata', 'generation_config']]
    
    print(f"✅ Loaded {len(categories_9bin)} 9-bin categories, {len(categories_25bin)} 25-bin categories")
    
    # Load dataset
    features, valence, arousal, audio_names = load_dataset(args.dataset_path)
    print(f"🎯 Testing on {min(args.num_samples, len(features))} samples")
    
    # Load model
    model = load_emotion_model(args.model_checkpoint, device)
    
    # Test all methods
    results = test_steering_method(model, features, valence, arousal, 
                                 steering_signals_9bin, steering_signals_25bin, 
                                 device, args.num_samples)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print_comparison(metrics)
    
    print(f"\n✅ **COMPREHENSIVE COMPARISON COMPLETE!**")
    print(f"   Methods tested: Baseline, 9-bin, 25-bin Categorical")
    print(f"   Samples processed: {len(results['targets']['valence'])}")


if __name__ == '__main__':
    main() 