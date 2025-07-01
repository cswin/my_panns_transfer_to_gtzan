#!/usr/bin/env python3
"""
Simple Interpolation Test for Steering Signals

This script tests interpolation-based steering signal selection using
the existing framework with minimal modifications.
"""

import numpy as np
import json
import os
import sys
import torch
from scipy.stats import pearsonr

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('src')

from data.data_generator import EmoSoundscapesDataset, EmotionValidateSampler, emotion_collate_fn
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
from utils.config import cnn6_config

def get_category_center(category_name):
    """Get the center coordinates of a 9-bin category."""
    parts = category_name.split('_')
    valence_label, arousal_label = parts[0], parts[1]
    
    valence_centers = {
        'negative': -0.665,    # Center of [-1.0, -0.33]
        'neutral': 0.0,        # Center of [-0.33, 0.33]  
        'positive': 0.665      # Center of [0.33, 1.0]
    }
    
    arousal_centers = {
        'weak': -0.665,        # Center of [-1.0, -0.33]
        'middle': 0.0,         # Center of [-0.33, 0.33]
        'strong': 0.665        # Center of [0.33, 1.0]
    }
    
    return valence_centers[valence_label], arousal_centers[arousal_label]

def select_steering_signal_by_target_original(steering_signals, target_valence, target_arousal):
    """Original categorical selection (from test_steering_signals.py)."""
    # Define thresholds for categorization
    valence_thresholds = [-0.33, 0.33]
    arousal_thresholds = [-0.33, 0.33]
    
    # Categorize valence
    if target_valence < valence_thresholds[0]:
        valence_category = "negative"
    elif target_valence > valence_thresholds[1]:
        valence_category = "positive"
    else:
        valence_category = "neutral"
    
    # Categorize arousal
    if target_arousal < arousal_thresholds[0]:
        arousal_category = "weak"
    elif target_arousal > arousal_thresholds[1]:
        arousal_category = "strong"
    else:
        arousal_category = "middle"
    
    # Construct category name
    category_name = f"{valence_category}_{arousal_category}"
    
    # Check if this category exists in steering signals
    if category_name in steering_signals:
        signals = steering_signals[category_name]
        if 'valence_128d' in signals and 'arousal_128d' in signals:
            valence_128d = np.array(signals['valence_128d'])
            arousal_128d = np.array(signals['arousal_128d'])
            return category_name, valence_128d, arousal_128d
    
    return None, None, None

def select_steering_signal_interpolated(steering_signals, target_valence, target_arousal, 
                                      k_neighbors=3, distance_power=2.0):
    """
    IMPROVED: Select steering signal using distance-weighted interpolation.
    """
    
    # Calculate distances to all category centers
    distances = {}
    for category_name in steering_signals.keys():
        if category_name in ['metadata', 'generation_config']:
            continue
        center_v, center_a = get_category_center(category_name)
        distance = np.sqrt((target_valence - center_v)**2 + (target_arousal - center_a)**2)
        distances[category_name] = distance
    
    # Select k nearest neighbors
    sorted_categories = sorted(distances.items(), key=lambda x: x[1])
    nearest_neighbors = sorted_categories[:k_neighbors]
    
    # Handle exact matches
    if nearest_neighbors[0][1] < 1e-8:
        category_name = nearest_neighbors[0][0]
        signals = steering_signals[category_name]
        valence_128d = np.array(signals['valence_128d'])
        arousal_128d = np.array(signals['arousal_128d'])
        return f"exact_{category_name}", valence_128d, arousal_128d
    
    # Calculate inverse distance weights
    weights = []
    categories = []
    
    for category_name, distance in nearest_neighbors:
        weight = 1.0 / (distance ** distance_power)
        weights.append(weight)
        categories.append(category_name)
    
    # Normalize weights
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    # Interpolate steering signals
    interpolated_valence = np.zeros(128)
    interpolated_arousal = np.zeros(128)
    
    for i, category_name in enumerate(categories):
        signals = steering_signals[category_name]
        valence_signal = np.array(signals['valence_128d'])
        arousal_signal = np.array(signals['arousal_128d'])
        
        interpolated_valence += weights[i] * valence_signal
        interpolated_arousal += weights[i] * arousal_signal
    
    method_name = f"interp_k{k_neighbors}_p{distance_power:.1f}"
    return method_name, interpolated_valence, interpolated_arousal

def test_steering_on_single_sample(model, feature, target_v, target_a, valence_signal, arousal_signal, 
                                  strength=1.0, device='cuda'):
    """Test steering on a single sample and return predictions."""
    model.eval()
    
    # Convert signals to tensors
    val_tensor = torch.tensor(valence_signal, dtype=torch.float32).to(device)
    aro_tensor = torch.tensor(arousal_signal, dtype=torch.float32).to(device)
    
    # Apply steering using external method (following tmp/ code pattern)
    model.add_steering_signal(
        source='affective_valence_128d',
        activation=val_tensor,
        strength=strength,
        alpha=1.0
    )
    model.add_steering_signal(
        source='affective_arousal_128d', 
        activation=aro_tensor,
        strength=strength,
        alpha=1.0
    )
    
    # Enable LRM
    model.lrm.enable()
    
    # Get prediction
    with torch.no_grad():
        output = model(feature, forward_passes=2)
        pred_v = output['valence'].cpu().numpy().flatten()[0]
        pred_a = output['arousal'].cpu().numpy().flatten()[0]
    
    # Clear state
    model.clear_feedback_state()
    
    return pred_v, pred_a

def compare_selection_methods():
    """Compare original vs interpolation selection methods."""
    print("🔄 INTERPOLATION vs CATEGORICAL STEERING COMPARISON")
    print("=" * 65)
    
    # Load model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000,
        window_size=cnn6_config['window_size'],
        hop_size=cnn6_config['hop_size'], 
        mel_bins=cnn6_config['mel_bins'], 
        fmin=cnn6_config['fmin'], 
        fmax=cnn6_config['fmax'],
        freeze_base=True,
        forward_passes=2
    )
    
    # Load trained weights
    checkpoint_path = 'workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"✅ Loaded model from {checkpoint_path}")
    else:
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Load steering signals
    steering_signals_path = 'tmp/steering_signals_by_category.json'
    if not os.path.exists(steering_signals_path):
        print(f"❌ Steering signals not found at {steering_signals_path}")
        print("   Please run scripts/generate_steering_signals.py first")
        return
    
    with open(steering_signals_path, 'r') as f:
        steering_signals = json.load(f)
    
    print(f"✅ Loaded {len([k for k in steering_signals.keys() if k not in ['metadata', 'generation_config']])} steering signal categories")
    
    # Create validation dataset and sampler
    dataset = EmoSoundscapesDataset()
    dataset_path = 'workspaces/emotion_feedback/features/emotion_features.h5'
    
    sampler = EmotionValidateSampler(
        hdf5_path=dataset_path,
        batch_size=1,
        train_ratio=0.7  # Use same split as training
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler, 
        collate_fn=emotion_collate_fn, num_workers=0
    )
    
    print(f"✅ Created validation dataset with {len(sampler.validate_indexes)} samples")
    
    # Test on subset of validation data
    results = {
        'categorical': {'valence_preds': [], 'arousal_preds': [], 'targets_v': [], 'targets_a': []},
        'interpolation': {'valence_preds': [], 'arousal_preds': [], 'targets_v': [], 'targets_a': []}
    }
    
    test_samples = 30  # Test on 30 samples for comparison
    sample_count = 0
    
    print(f"\n🎯 Testing on {test_samples} validation samples")
    print("\nSample-by-sample comparison:")
    print("Sample | Target V,A | Categorical | Interpolated | V Diff | A Diff")
    print("-" * 70)
    
    for batch_meta in sampler:
        if sample_count >= test_samples:
            break
        
        # Process single sample (batch_size=1)
        meta = batch_meta[0]  # Get first (and only) item from batch
        sample_data = dataset[meta]
        
        # Extract data
        feature = torch.tensor(sample_data['feature']).unsqueeze(0).to(device)  # Add batch dimension
        target_v = sample_data['valence']
        target_a = sample_data['arousal']
        
        # Test categorical method
        cat_category, val_signal_cat, aro_signal_cat = select_steering_signal_by_target_original(
            steering_signals, target_v, target_a
        )
        
        if val_signal_cat is not None:
            pred_v_cat, pred_a_cat = test_steering_on_single_sample(
                model, feature, target_v, target_a, val_signal_cat, aro_signal_cat, 
                strength=1.0, device=device
            )
        else:
            pred_v_cat, pred_a_cat = 0.0, 0.0
            cat_category = "none"
        
        # Test interpolation method
        interp_method, val_signal_interp, aro_signal_interp = select_steering_signal_interpolated(
            steering_signals, target_v, target_a, k_neighbors=3, distance_power=2.0
        )
        
        pred_v_interp, pred_a_interp = test_steering_on_single_sample(
            model, feature, target_v, target_a, val_signal_interp, aro_signal_interp,
            strength=1.0, device=device
        )
        
        # Calculate differences (negative = interpolation better)
        v_diff = abs(pred_v_interp - target_v) - abs(pred_v_cat - target_v)
        a_diff = abs(pred_a_interp - target_a) - abs(pred_a_cat - target_a)
        
        # Store results
        results['categorical']['valence_preds'].append(pred_v_cat)
        results['categorical']['arousal_preds'].append(pred_a_cat)
        results['categorical']['targets_v'].append(target_v)
        results['categorical']['targets_a'].append(target_a)
        
        results['interpolation']['valence_preds'].append(pred_v_interp)
        results['interpolation']['arousal_preds'].append(pred_a_interp)
        results['interpolation']['targets_v'].append(target_v)
        results['interpolation']['targets_a'].append(target_a)
        
        # Display comparison for first 10 samples
        if sample_count < 10:
            v_diff_str = f"{v_diff:+.3f}" if abs(v_diff) > 0.001 else "~0.000"
            a_diff_str = f"{a_diff:+.3f}" if abs(a_diff) > 0.001 else "~0.000"
            
            print(f"{sample_count:6d} | {target_v:+.2f},{target_a:+.2f} | "
                  f"{cat_category[:12]:12} | {interp_method[:12]:12} | {v_diff_str:6s} | {a_diff_str:6s}")
        
        sample_count += 1
    
    return results

def calculate_performance_metrics(results):
    """Calculate and compare performance metrics."""
    print(f"\n📈 PERFORMANCE COMPARISON RESULTS")
    print("=" * 50)
    
    for method_name, data in results.items():
        targets_v = np.array(data['targets_v'])
        targets_a = np.array(data['targets_a'])
        preds_v = np.array(data['valence_preds'])
        preds_a = np.array(data['arousal_preds'])
        
        # Calculate correlations
        r_v, _ = pearsonr(targets_v, preds_v)
        r_a, _ = pearsonr(targets_a, preds_a)
        
        # Calculate RMSE
        rmse_v = np.sqrt(np.mean((targets_v - preds_v) ** 2))
        rmse_a = np.sqrt(np.mean((targets_a - preds_a) ** 2))
        
        # Calculate MAE
        mae_v = np.mean(np.abs(targets_v - preds_v))
        mae_a = np.mean(np.abs(targets_a - preds_a))
        
        print(f"\n🎯 **{method_name.upper()} METHOD:**")
        print(f"   Valence:  r={r_v:.3f}, RMSE={rmse_v:.3f}, MAE={mae_v:.3f}")
        print(f"   Arousal:  r={r_a:.3f}, RMSE={rmse_a:.3f}, MAE={mae_a:.3f}")
    
    # Calculate improvements
    cat_r_v = pearsonr(results['categorical']['targets_v'], results['categorical']['valence_preds'])[0]
    cat_r_a = pearsonr(results['categorical']['targets_a'], results['categorical']['arousal_preds'])[0]
    
    interp_r_v = pearsonr(results['interpolation']['targets_v'], results['interpolation']['valence_preds'])[0]
    interp_r_a = pearsonr(results['interpolation']['targets_a'], results['interpolation']['arousal_preds'])[0]
    
    print(f"\n🚀 **IMPROVEMENT ANALYSIS:**")
    print(f"   Valence correlation: {cat_r_v:.3f} → {interp_r_v:.3f} (Δr = {interp_r_v - cat_r_v:+.3f})")
    print(f"   Arousal correlation: {cat_r_a:.3f} → {interp_r_a:.3f} (Δr = {interp_r_a - cat_r_a:+.3f})")
    
    if interp_r_v > cat_r_v and interp_r_a > cat_r_a:
        print("   ✅ **INTERPOLATION WINS**: Better performance on both dimensions")
    elif interp_r_v > cat_r_v or interp_r_a > cat_r_a:
        print("   🔄 **MIXED RESULTS**: Interpolation better on one dimension")
    else:
        print("   ⚠️  **CATEGORICAL WINS**: Original method still better")
    
    return {
        'categorical': {'r_v': cat_r_v, 'r_a': cat_r_a},
        'interpolation': {'r_v': interp_r_v, 'r_a': interp_r_a}
    }

def main():
    """Main testing function."""
    print("🧪 SIMPLE INTERPOLATION STEERING TEST")
    print("=" * 50)
    
    # Compare selection methods
    results = compare_selection_methods()
    
    if results:
        # Calculate performance metrics
        metrics = calculate_performance_metrics(results)
        
        print(f"\n✅ **INTERPOLATION TEST COMPLETE!**")
        print(f"\n🎯 **KEY FINDINGS:**")
        print(f"   - Interpolation provides distance-weighted steering signal blending")
        print(f"   - Uses existing 9-bin signals without regeneration")
        print(f"   - Reduces boundary sensitivity through smooth transitions")
        print(f"   - Performance change: Valence Δr = {metrics['interpolation']['r_v'] - metrics['categorical']['r_v']:+.3f}, Arousal Δr = {metrics['interpolation']['r_a'] - metrics['categorical']['r_a']:+.3f}")
        
        print(f"\n🚀 **NEXT STEPS:**")
        print(f"   1. If interpolation shows improvement, test with different strengths")
        print(f"   2. Try different interpolation parameters (k_neighbors, distance_power)")
        print(f"   3. Test on larger validation set for statistical significance")
        print(f"   4. Consider combining with finer categorization (25-bin system)")

if __name__ == '__main__':
    main() 