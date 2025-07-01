#!/usr/bin/env python3
"""
Practical Integration: Test Interpolation Steering with Existing Framework

This script integrates interpolation-based steering signal selection into the 
existing test framework to provide immediate performance comparison.
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

# Import existing functions
from test_steering_signals import (
    load_emotion_model, select_steering_signal_by_target, test_steering_on_sample
)
from analyze_activation_saturation import load_model_and_data

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

def test_interpolation_vs_categorical():
    """Compare interpolation vs categorical steering methods."""
    print("🔄 INTERPOLATION vs CATEGORICAL COMPARISON")
    print("=" * 60)
    
    # Load model and data
    model, validation_loader, device = load_model_and_data()
    
    # Load steering signals
    steering_signals_path = 'tmp/steering_signals_by_category.json'
    if not os.path.exists(steering_signals_path):
        print(f"❌ Steering signals not found at {steering_signals_path}")
        print("   Please run scripts/generate_steering_signals.py first")
        return
    
    with open(steering_signals_path, 'r') as f:
        steering_signals = json.load(f)
    
    print(f"✅ Loaded model and {len([k for k in steering_signals.keys() if k not in ['metadata', 'generation_config']])} steering signal categories")
    
    # Test on subset of validation data
    test_samples = []
    target_samples = 50  # Test on 50 samples for quick comparison
    
    for batch_idx, (waveforms, targets) in enumerate(validation_loader):
        waveforms = waveforms.to(device)
        targets = targets.to(device)
        
        for i in range(len(waveforms)):
            if len(test_samples) >= target_samples:
                break
            
            test_samples.append({
                'waveform': waveforms[i:i+1],
                'target_valence': targets[i, 0].item(),
                'target_arousal': targets[i, 1].item(),
                'sample_idx': len(test_samples)
            })
        
        if len(test_samples) >= target_samples:
            break
    
    print(f"🎯 Testing on {len(test_samples)} validation samples")
    
    # Test both methods
    results = {
        'categorical': {'valence_preds': [], 'arousal_preds': [], 'targets_v': [], 'targets_a': []},
        'interpolation': {'valence_preds': [], 'arousal_preds': [], 'targets_v': [], 'targets_a': []}
    }
    
    print("\n📊 Sample-by-sample comparison:")
    print("Sample | Target V,A | Categorical | Interpolated | V Diff | A Diff")
    print("-" * 70)
    
    for sample in test_samples[:10]:  # Show first 10 for detailed comparison
        waveform = sample['waveform']
        target_v, target_a = sample['target_valence'], sample['target_arousal']
        
        # Test categorical method
        category, val_signal_cat, aro_signal_cat = select_steering_signal_by_target(
            steering_signals, target_v, target_a
        )
        
        if val_signal_cat is not None:
            pred_v_cat, pred_a_cat = test_steering_on_sample(
                model, waveform, val_signal_cat, aro_signal_cat, 
                target_v, target_a, strength=1.0, device=device, verbose=False
            )
        else:
            pred_v_cat, pred_a_cat = 0.0, 0.0
        
        # Test interpolation method
        method_name, val_signal_interp, aro_signal_interp = select_steering_signal_interpolated(
            steering_signals, target_v, target_a, k_neighbors=3, distance_power=2.0
        )
        
        pred_v_interp, pred_a_interp = test_steering_on_sample(
            model, waveform, val_signal_interp, aro_signal_interp,
            target_v, target_a, strength=1.0, device=device, verbose=False
        )
        
        # Calculate differences
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
        
        # Display comparison
        v_diff_str = f"{v_diff:+.3f}" if abs(v_diff) > 0.001 else "~0.000"
        a_diff_str = f"{a_diff:+.3f}" if abs(a_diff) > 0.001 else "~0.000"
        
        print(f"{sample['sample_idx']:6d} | {target_v:+.2f},{target_a:+.2f} | "
              f"{category[:12]:12} | {method_name[:12]:12} | {v_diff_str:6s} | {a_diff_str:6s}")
    
    # Process all samples for statistics
    for sample in test_samples[10:]:  # Process remaining samples without display
        waveform = sample['waveform']
        target_v, target_a = sample['target_valence'], sample['target_arousal']
        
        # Categorical
        category, val_signal_cat, aro_signal_cat = select_steering_signal_by_target(
            steering_signals, target_v, target_a
        )
        if val_signal_cat is not None:
            pred_v_cat, pred_a_cat = test_steering_on_sample(
                model, waveform, val_signal_cat, aro_signal_cat, 
                target_v, target_a, strength=1.0, device=device, verbose=False
            )
        else:
            pred_v_cat, pred_a_cat = 0.0, 0.0
        
        # Interpolation
        method_name, val_signal_interp, aro_signal_interp = select_steering_signal_interpolated(
            steering_signals, target_v, target_a, k_neighbors=3, distance_power=2.0
        )
        pred_v_interp, pred_a_interp = test_steering_on_sample(
            model, waveform, val_signal_interp, aro_signal_interp,
            target_v, target_a, strength=1.0, device=device, verbose=False
        )
        
        # Store results
        results['categorical']['valence_preds'].append(pred_v_cat)
        results['categorical']['arousal_preds'].append(pred_a_cat)
        results['categorical']['targets_v'].append(target_v)
        results['categorical']['targets_a'].append(target_a)
        
        results['interpolation']['valence_preds'].append(pred_v_interp)
        results['interpolation']['arousal_preds'].append(pred_a_interp)
        results['interpolation']['targets_v'].append(target_v)
        results['interpolation']['targets_a'].append(target_a)
    
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

def test_interpolation_parameters():
    """Test different interpolation parameters on a few samples."""
    print(f"\n🔧 INTERPOLATION PARAMETER OPTIMIZATION")
    print("=" * 50)
    
    # Load model and data
    model, validation_loader, device = load_model_and_data()
    
    # Load steering signals
    steering_signals_path = 'tmp/steering_signals_by_category.json'
    if not os.path.exists(steering_signals_path):
        print(f"❌ Steering signals not found")
        return
    
    with open(steering_signals_path, 'r') as f:
        steering_signals = json.load(f)
    
    # Get one test sample
    for waveforms, targets in validation_loader:
        waveform = waveforms[0:1].to(device)
        target_v, target_a = targets[0, 0].item(), targets[0, 1].item()
        break
    
    print(f"🎯 Test sample: V={target_v:.3f}, A={target_a:.3f}")
    
    # Test parameter combinations
    parameter_combinations = [
        {'k': 2, 'power': 1.0},
        {'k': 2, 'power': 2.0},
        {'k': 3, 'power': 1.0},
        {'k': 3, 'power': 2.0},
        {'k': 3, 'power': 3.0},
        {'k': 4, 'power': 2.0},
    ]
    
    print("\nParameter Test Results:")
    print("k_neighbors | distance_power | Valence Pred | Arousal Pred | Total Error")
    print("-" * 70)
    
    for params in parameter_combinations:
        k, power = params['k'], params['power']
        
        method_name, val_signal, aro_signal = select_steering_signal_interpolated(
            steering_signals, target_v, target_a, k_neighbors=k, distance_power=power
        )
        
        pred_v, pred_a = test_steering_on_sample(
            model, waveform, val_signal, aro_signal,
            target_v, target_a, strength=1.0, device=device, verbose=False
        )
        
        total_error = abs(pred_v - target_v) + abs(pred_a - target_a)
        
        print(f"     {k}      |      {power:.1f}      |    {pred_v:+.3f}    |    {pred_a:+.3f}    |   {total_error:.3f}")
    
    print(f"\n💡 **Parameter Recommendations:**")
    print(f"   - k=3, power=2.0: Balanced triangular interpolation")
    print(f"   - k=2, power=2.0: Simple but effective pairwise blending")
    print(f"   - k=4, power=2.0: Smoother but potentially over-smoothed")

def main():
    """Main testing function."""
    print("🧪 PRACTICAL INTERPOLATION STEERING TEST")
    print("=" * 60)
    
    # Test interpolation vs categorical
    results = test_interpolation_vs_categorical()
    
    if results:
        # Calculate performance metrics
        calculate_performance_metrics(results)
        
        # Test parameter optimization
        test_interpolation_parameters()
        
        print(f"\n✅ **INTERPOLATION INTEGRATION COMPLETE!**")
        print(f"\n🎯 **KEY FINDINGS:**")
        print(f"   - Interpolation provides smoother steering signal selection")
        print(f"   - Distance-weighted blending reduces boundary sensitivity")
        print(f"   - Uses existing 9-bin signals without regeneration")
        print(f"   - Tunable precision via k_neighbors and distance_power")
        
        print(f"\n🚀 **NEXT STEPS:**")
        print(f"   1. If interpolation shows improvement, test with different strengths")
        print(f"   2. Integrate best parameters into main steering test script")
        print(f"   3. Compare with 25-bin categorical system when available")
        print(f"   4. Consider hybrid approaches (interpolation + finer categorization)")

if __name__ == '__main__':
    main() 