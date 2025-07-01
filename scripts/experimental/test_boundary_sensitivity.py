#!/usr/bin/env python3
"""
Test Boundary Sensitivity: Categorical vs Interpolation

This script specifically tests scenarios where interpolation should outperform
categorical selection, focusing on targets near category boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

def analyze_boundary_sensitivity():
    """Analyze how categorical vs interpolation methods handle boundary cases."""
    print("🔍 BOUNDARY SENSITIVITY ANALYSIS")
    print("=" * 50)
    
    # Load steering signals
    steering_signals_path = 'tmp/steering_signals_by_category.json'
    if not os.path.exists(steering_signals_path):
        print(f"❌ Steering signals not found at {steering_signals_path}")
        return
    
    with open(steering_signals_path, 'r') as f:
        steering_signals = json.load(f)
    
    print(f"✅ Loaded {len([k for k in steering_signals.keys() if k not in ['metadata', 'generation_config']])} steering signal categories")
    
    # Define boundary test cases
    boundary_cases = [
        # Cases right at category boundaries (most sensitive)
        {'v': -0.33, 'a': 0.0, 'desc': 'Valence boundary: negative/neutral'},
        {'v': 0.33, 'a': 0.0, 'desc': 'Valence boundary: neutral/positive'},
        {'v': 0.0, 'a': -0.33, 'desc': 'Arousal boundary: weak/middle'},
        {'v': 0.0, 'a': 0.33, 'desc': 'Arousal boundary: middle/strong'},
        
        # Cases just inside boundaries (should be stable)
        {'v': -0.35, 'a': 0.0, 'desc': 'Just inside negative'},
        {'v': 0.35, 'a': 0.0, 'desc': 'Just inside positive'},
        {'v': 0.0, 'a': -0.35, 'desc': 'Just inside weak'},
        {'v': 0.0, 'a': 0.35, 'desc': 'Just inside strong'},
        
        # Corner cases (multiple boundaries)
        {'v': 0.33, 'a': 0.33, 'desc': 'Corner: neutral/positive, middle/strong'},
        {'v': -0.33, 'a': -0.33, 'desc': 'Corner: negative/neutral, weak/middle'},
        
        # Cases far from any boundary (should be identical)
        {'v': 0.0, 'a': 0.0, 'desc': 'Center of neutral_middle'},
        {'v': -0.665, 'a': 0.665, 'desc': 'Center of negative_strong'},
    ]
    
    print(f"\n📊 Testing {len(boundary_cases)} boundary sensitivity cases:")
    print("Target Emotion | Description | Categorical | Interpolation | Difference")
    print("-" * 80)
    
    categorical_selections = []
    interpolation_selections = []
    differences = []
    
    for case in boundary_cases:
        v, a = case['v'], case['a']
        desc = case['desc']
        
        # Categorical selection
        cat_category, cat_val, cat_aro = select_steering_signal_by_target_original(
            steering_signals, v, a
        )
        
        # Interpolation selection
        interp_method, interp_val, interp_aro = select_steering_signal_interpolated(
            steering_signals, v, a, k_neighbors=3, distance_power=2.0
        )
        
        # Calculate signal differences
        if cat_val is not None and interp_val is not None:
            val_diff = np.mean(np.abs(np.array(interp_val) - np.array(cat_val)))
            aro_diff = np.mean(np.abs(np.array(interp_aro) - np.array(cat_aro)))
            total_diff = val_diff + aro_diff
        else:
            total_diff = 0.0
        
        categorical_selections.append(cat_category)
        interpolation_selections.append(interp_method)
        differences.append(total_diff)
        
        print(f"V={v:+.2f}, A={a:+.2f} | {desc[:25]:25} | {cat_category:12} | {interp_method:12} | {total_diff:.4f}")
    
    return boundary_cases, categorical_selections, interpolation_selections, differences

def create_boundary_visualization():
    """Create visualization showing boundary sensitivity effects."""
    print(f"\n📊 Creating boundary sensitivity visualization...")
    
    # Create a fine grid around boundaries
    valence_range = np.linspace(-0.5, 0.5, 100)
    arousal_range = np.linspace(-0.5, 0.5, 100)
    V, A = np.meshgrid(valence_range, arousal_range)
    
    # Load steering signals
    steering_signals_path = 'tmp/steering_signals_by_category.json'
    with open(steering_signals_path, 'r') as f:
        steering_signals = json.load(f)
    
    # Calculate differences across the grid
    differences = np.zeros_like(V)
    
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v, a = V[i, j], A[i, j]
            
            # Categorical selection
            cat_category, cat_val, cat_aro = select_steering_signal_by_target_original(
                steering_signals, v, a
            )
            
            # Interpolation selection
            interp_method, interp_val, interp_aro = select_steering_signal_interpolated(
                steering_signals, v, a, k_neighbors=3, distance_power=2.0
            )
            
            # Calculate difference
            if cat_val is not None and interp_val is not None:
                val_diff = np.mean(np.abs(np.array(interp_val) - np.array(cat_val)))
                aro_diff = np.mean(np.abs(np.array(interp_aro) - np.array(cat_aro)))
                differences[i, j] = val_diff + aro_diff
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Difference heatmap
    im1 = ax1.imshow(differences, extent=[-0.5, 0.5, -0.5, 0.5], origin='lower', 
                     cmap='viridis', aspect='equal')
    ax1.axvline(-0.33, color='red', linewidth=2, alpha=0.7, label='Category boundaries')
    ax1.axvline(0.33, color='red', linewidth=2, alpha=0.7)
    ax1.axhline(-0.33, color='red', linewidth=2, alpha=0.7)
    ax1.axhline(0.33, color='red', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Valence')
    ax1.set_ylabel('Arousal')
    ax1.set_title('Interpolation vs Categorical Signal Differences')
    ax1.legend()
    plt.colorbar(im1, ax=ax1, label='Total Signal Difference')
    
    # Plot 2: Boundary effect profile
    # Take a slice through the valence boundary at arousal=0
    boundary_slice = differences[50, :]  # Middle row (arousal ≈ 0)
    ax2.plot(valence_range, boundary_slice, 'b-', linewidth=2, label='Signal difference')
    ax2.axvline(-0.33, color='red', linestyle='--', alpha=0.7, label='Negative/Neutral boundary')
    ax2.axvline(0.33, color='red', linestyle='--', alpha=0.7, label='Neutral/Positive boundary')
    ax2.set_xlabel('Valence (at Arousal=0)')
    ax2.set_ylabel('Signal Difference')
    ax2.set_title('Boundary Effect Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tmp/boundary_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Boundary sensitivity visualization saved to tmp/boundary_sensitivity_analysis.png")
    
    return differences

def analyze_interpolation_parameters():
    """Analyze how different interpolation parameters affect boundary sensitivity."""
    print(f"\n🔧 INTERPOLATION PARAMETER ANALYSIS")
    print("=" * 50)
    
    # Load steering signals
    steering_signals_path = 'tmp/steering_signals_by_category.json'
    with open(steering_signals_path, 'r') as f:
        steering_signals = json.load(f)
    
    # Test case: right at boundary
    test_v, test_a = -0.33, 0.0  # Exactly at negative/neutral boundary
    
    print(f"🎯 Test case: V={test_v}, A={test_a} (at negative/neutral boundary)")
    print("\nParameter combinations and their effects:")
    print("k_neighbors | distance_power | Selected Categories | Weights")
    print("-" * 65)
    
    parameter_combinations = [
        {'k': 2, 'power': 1.0},
        {'k': 2, 'power': 2.0},
        {'k': 3, 'power': 1.0},
        {'k': 3, 'power': 2.0},
        {'k': 3, 'power': 3.0},
        {'k': 4, 'power': 2.0},
    ]
    
    for params in parameter_combinations:
        k, power = params['k'], params['power']
        
        # Get category distances
        distances = {}
        for category_name in steering_signals.keys():
            if category_name in ['metadata', 'generation_config']:
                continue
            center_v, center_a = get_category_center(category_name)
            distance = np.sqrt((test_v - center_v)**2 + (test_a - center_a)**2)
            distances[category_name] = distance
        
        # Select k nearest neighbors
        sorted_categories = sorted(distances.items(), key=lambda x: x[1])
        nearest_neighbors = sorted_categories[:k]
        
        # Calculate weights
        weights = []
        categories = []
        for category_name, distance in nearest_neighbors:
            weight = 1.0 / (distance ** power)
            weights.append(weight)
            categories.append(category_name)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Format output
        cat_str = ", ".join(categories)
        weight_str = ", ".join([f"{w:.3f}" for w in weights])
        
        print(f"     {k}      |      {power:.1f}      | {cat_str[:25]:25} | {weight_str}")
    
    print(f"\n💡 **Observations:**")
    print(f"   - Higher k includes more distant categories")
    print(f"   - Higher power concentrates weights on nearest neighbors")
    print(f"   - At boundaries, interpolation should blend adjacent categories")

# Import the selection functions from the interpolation test
import sys
sys.path.append('.')
from test_simple_interpolation import (
    get_category_center, 
    select_steering_signal_by_target_original,
    select_steering_signal_interpolated
)

def main():
    """Main analysis function."""
    print("🔍 BOUNDARY SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    # Analyze boundary sensitivity
    boundary_cases, cat_selections, interp_selections, differences = analyze_boundary_sensitivity()
    
    # Create visualization
    diff_grid = create_boundary_visualization()
    
    # Analyze interpolation parameters
    analyze_interpolation_parameters()
    
    print(f"\n✅ **BOUNDARY SENSITIVITY ANALYSIS COMPLETE!**")
    print(f"\n🎯 **KEY FINDINGS:**")
    
    # Analyze results
    max_diff = max(differences)
    avg_diff = np.mean(differences)
    boundary_diffs = differences[:4]  # First 4 are exact boundary cases
    center_diffs = differences[-2:]   # Last 2 are center cases
    
    print(f"   - Maximum signal difference: {max_diff:.4f}")
    print(f"   - Average signal difference: {avg_diff:.4f}")
    print(f"   - Boundary cases avg: {np.mean(boundary_diffs):.4f}")
    print(f"   - Center cases avg: {np.mean(center_diffs):.4f}")
    
    if max_diff > 0.001:
        print(f"   ✅ **INTERPOLATION IS WORKING**: Signals differ from categorical")
        print(f"   📊 **CHECK VISUALIZATION**: tmp/boundary_sensitivity_analysis.png")
    else:
        print(f"   ⚠️  **MINIMAL DIFFERENCES**: Interpolation very similar to categorical")
    
    print(f"\n🚀 **IMPLICATIONS:**")
    print(f"   - If differences are small: 9-bin precision may be the limiting factor")
    print(f"   - If differences are large at boundaries: Interpolation provides smoother transitions")
    print(f"   - This supports the need for finer categorization (25-bin system)")

if __name__ == '__main__':
    main() 