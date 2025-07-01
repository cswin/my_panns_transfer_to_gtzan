#!/usr/bin/env python3
"""
Test 25-bin steering signals: Categorical vs Interpolation comparison.
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


def categorize_25bin_with_fallback(valence, arousal):
    """Categorize emotions into 25 bins with smart fallback."""
    # 5x5 grid: very_negative, negative, neutral, positive, very_positive
    #           × very_weak, weak, middle, strong, very_strong
    
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


def select_categorical_25bin(target_valence, target_arousal, steering_signals):
    """Select steering signal using direct 25-bin categorization."""
    category = categorize_25bin_with_fallback(target_valence, target_arousal)
    
    if category in steering_signals:
        return steering_signals[category], category
    else:
        return None, None


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
    
    # Blend information
    blend_info = {
        'nearest_categories': nearest_categories,
        'distances': nearest_distances.tolist(),
        'weights': weights.tolist(),
        'k_neighbors': k_neighbors
    }
    
    return interpolated_signals, blend_info


def load_emotion_model(checkpoint_path, device):
    """Load the emotion model from checkpoint."""
    print(f"�� Loading model from {checkpoint_path}")
    
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
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model


def load_dataset(dataset_path, train_ratio=0.7, random_seed=42):
    """Load and split dataset."""
    print(f"📂 Loading dataset from {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as hf:
        features = hf['features'][:]
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


def main():
    """Main testing function."""
    print("🚀 25-BIN STEERING METHODS COMPARISON")
    print("=" * 60)
    
    # Quick test with 10 samples
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Load steering signals
    steering_path = 'tmp/25bin_steering_signals/steering_signals_25bin.json'
    print(f"📂 Loading 25-bin steering signals from {steering_path}")
    with open(steering_path, 'r') as f:
        steering_signals = json.load(f)
    
    available_categories = [k for k in steering_signals.keys() if k not in ['metadata', 'generation_config']]
    print(f"✅ Loaded {len(available_categories)} steering signal categories")
    
    # Test categorization
    test_points = [
        (-0.8, -0.8, "very_negative_very_weak"),
        (-0.4, 0.0, "negative_middle"),
        (0.0, 0.0, "neutral_middle"),
        (0.4, 0.4, "positive_strong"),
        (0.8, 0.8, "very_positive_very_strong")
    ]
    
    print(f"\n🧪 Testing categorization:")
    for v, a, expected in test_points:
        categorical = categorize_25bin_with_fallback(v, a)
        print(f"  V={v:+.1f}, A={a:+.1f} → {categorical}")
        
        # Test interpolation
        interpolated, blend_info = select_interpolated_25bin(v, a, steering_signals)
        if interpolated:
            print(f"    Interpolation: {blend_info['k_neighbors']} neighbors, weights: {[f'{w:.3f}' for w in blend_info['weights']]}")
    
    print(f"\n✅ **25-BIN TEST COMPLETE!**")
    print(f"   Categories available: {len(available_categories)}")


if __name__ == '__main__':
    main()
