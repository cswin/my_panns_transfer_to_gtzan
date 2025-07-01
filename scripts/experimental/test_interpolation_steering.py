#!/usr/bin/env python3
"""
Simple Interpolation-Based Steering Signal Improvement

This script implements a distance-weighted interpolation approach that blends
neighboring steering signals based on target proximity, using existing 9-bin signals.
This provides better precision without requiring new signal generation.
"""

import numpy as np
import json
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('src')

def get_category_center(category_name):
    """Get the center coordinates of a 9-bin category."""
    
    # Parse category name (e.g., "positive_strong")
    parts = category_name.split('_')
    valence_label, arousal_label = parts[0], parts[1]
    
    # Map to center coordinates
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

def calculate_category_distances(target_valence, target_arousal, steering_signals):
    """Calculate distances from target to all category centers."""
    
    distances = {}
    
    for category_name in steering_signals.keys():
        if category_name in ['metadata', 'generation_config']:
            continue
            
        center_v, center_a = get_category_center(category_name)
        
        # Euclidean distance to category center
        distance = np.sqrt((target_valence - center_v)**2 + (target_arousal - center_a)**2)
        distances[category_name] = distance
    
    return distances

def select_steering_signal_interpolated(steering_signals, target_valence, target_arousal, 
                                      k_neighbors=3, distance_power=2.0):
    """
    IMPROVED: Select steering signal using distance-weighted interpolation.
    
    Args:
        steering_signals: dict, existing 9-bin steering signals
        target_valence: float, target valence value
        target_arousal: float, target arousal value  
        k_neighbors: int, number of nearest neighbors to blend
        distance_power: float, power for distance weighting (higher = more local)
    
    Returns:
        tuple: (method_name, interpolated_valence_128d, interpolated_arousal_128d)
    """
    
    # Calculate distances to all category centers
    distances = calculate_category_distances(target_valence, target_arousal, steering_signals)
    
    # Sort by distance and select k nearest neighbors
    sorted_categories = sorted(distances.items(), key=lambda x: x[1])
    nearest_neighbors = sorted_categories[:k_neighbors]
    
    # Calculate inverse distance weights
    weights = []
    categories = []
    
    for category_name, distance in nearest_neighbors:
        # Avoid division by zero for exact matches
        if distance < 1e-8:
            # If we're exactly at a category center, use only that category
            signals = steering_signals[category_name]
            valence_128d = np.array(signals['valence_128d'])
            arousal_128d = np.array(signals['arousal_128d'])
            return f"exact_{category_name}", valence_128d, arousal_128d
        
        # Inverse distance weighting
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
    
    # Create method description
    method_name = f"interp_k{k_neighbors}_p{distance_power:.1f}"
    
    return method_name, interpolated_valence, interpolated_arousal

def compare_selection_methods():
    """Compare original categorical vs interpolation methods."""
    print("🔄 STEERING SIGNAL SELECTION METHOD COMPARISON")
    print("=" * 65)
    
    # Load existing steering signals
    steering_signals_path = 'tmp/steering_signals_by_category.json'
    
    if not os.path.exists(steering_signals_path):
        print(f"❌ Steering signals not found at {steering_signals_path}")
        print("   Please run scripts/generate_steering_signals.py first")
        return
    
    with open(steering_signals_path, 'r') as f:
        steering_signals = json.load(f)
    
    print(f"✅ Loaded steering signals with {len([k for k in steering_signals.keys() if k not in ['metadata', 'generation_config']])} categories")
    
    # Test cases with different proximity to category centers
    test_cases = [
        {'v': 0.0, 'a': 0.0, 'desc': 'Exact center (neutral_middle)'},
        {'v': -0.665, 'a': 0.665, 'desc': 'Exact center (negative_strong)'},
        {'v': -0.1, 'a': 0.1, 'desc': 'Near center, slight offset'},
        {'v': -0.5, 'a': 0.2, 'desc': 'Between categories'},
        {'v': 0.33, 'a': 0.33, 'desc': 'Exactly on boundary'},
        {'v': 0.5, 'a': -0.5, 'desc': 'Far from any center'},
    ]
    
    print("\n📊 METHOD COMPARISON:")
    print("Target Emotion → Original Category → Interpolation Method → Neighbors Used")
    print("-" * 85)
    
    for case in test_cases:
        v, a = case['v'], case['a']
        
        # Original categorical method
        from test_steering_signals import select_steering_signal_by_target
        original_category, _, _ = select_steering_signal_by_target(steering_signals, v, a)
        
        # Interpolation method
        interp_method, _, _ = select_steering_signal_interpolated(steering_signals, v, a, k_neighbors=3)
        
        # Get neighbor info
        distances = calculate_category_distances(v, a, steering_signals)
        nearest_3 = sorted(distances.items(), key=lambda x: x[1])[:3]
        neighbor_names = [name for name, dist in nearest_3]
        
        print(f"V={v:5.2f}, A={a:5.2f} → {original_category:15} → {interp_method:20} → {neighbor_names}")
    
    return steering_signals

def test_interpolation_parameters():
    """Test different interpolation parameters."""
    print("\n🔧 INTERPOLATION PARAMETER TESTING")
    print("=" * 50)
    
    # Load steering signals
    steering_signals_path = 'tmp/steering_signals_by_category.json'
    
    if not os.path.exists(steering_signals_path):
        print(f"❌ Steering signals not found")
        return
    
    with open(steering_signals_path, 'r') as f:
        steering_signals = json.load(f)
    
    # Test target between categories
    target_v, target_a = -0.1, 0.1
    
    print(f"🎯 Test Target: V={target_v}, A={target_a} (between neutral_middle and neighbors)")
    print("\nParameter Variations:")
    print("k_neighbors | distance_power | Method Name | Top 3 Weights")
    print("-" * 60)
    
    # Test different parameter combinations
    for k in [2, 3, 4]:
        for power in [1.0, 2.0, 3.0]:
            method_name, val_signal, aro_signal = select_steering_signal_interpolated(
                steering_signals, target_v, target_a, k_neighbors=k, distance_power=power
            )
            
            # Calculate weights for display
            distances = calculate_category_distances(target_v, target_a, steering_signals)
            nearest_k = sorted(distances.items(), key=lambda x: x[1])[:k]
            
            weights = []
            for _, distance in nearest_k:
                weight = 1.0 / (distance ** power)
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            weight_str = ", ".join([f"{w:.3f}" for w in weights[:3]])
            print(f"     {k}      |      {power:.1f}      | {method_name:15} | {weight_str}")

def analyze_interpolation_benefits():
    """Analyze theoretical benefits of interpolation approach."""
    print("\n💡 INTERPOLATION BENEFITS ANALYSIS")
    print("=" * 50)
    
    print("🎯 **Advantages over Categorical Selection:**")
    print("1. **Smooth transitions**: No abrupt changes at category boundaries")
    print("2. **Better target matching**: Weighted blend closer to actual target")
    print("3. **Reduced boundary sensitivity**: Gradual weight changes")
    print("4. **No new signal generation**: Uses existing 9-bin signals")
    print("5. **Tunable precision**: Adjust k_neighbors and distance_power")
    
    print("\n🔧 **Parameter Effects:**")
    print("- **k_neighbors=2**: Simple linear interpolation between closest categories")
    print("- **k_neighbors=3**: Triangular interpolation, smoother blending")
    print("- **k_neighbors=4+**: More global influence, potentially smoother")
    print("- **distance_power=1.0**: Linear distance weighting")
    print("- **distance_power=2.0**: Quadratic weighting (more local)")
    print("- **distance_power=3.0+**: Very local weighting (sharp transitions)")
    
    print("\n📈 **Expected Performance Improvements:**")
    print("- **Better target approximation**: Weighted average closer to target than category center")
    print("- **Reduced approximation error**: Especially for targets between categories")
    print("- **Potential for higher strengths**: Better precision may allow amplification")
    print("- **Maintained robustness**: Still uses proven steering signal components")
    
    print("\n🧪 **Recommended Testing Strategy:**")
    print("1. **Baseline**: Test k=3, power=2.0 (balanced interpolation)")
    print("2. **Compare strengths**: Test 1.0x, 1.2x, 1.5x with interpolation")
    print("3. **Parameter sweep**: Find optimal k and power combination")
    print("4. **Boundary analysis**: Check performance near category boundaries")

def generate_interpolation_implementation(output_dir='tmp'):
    """Generate implementation code for interpolated steering."""
    os.makedirs(output_dir, exist_ok=True)
    
    code = '''
def select_steering_signal_interpolated(steering_signals, target_valence, target_arousal, 
                                      k_neighbors=3, distance_power=2.0):
    """
    IMPROVED: Select steering signal using distance-weighted interpolation.
    
    This method blends k nearest category signals based on distance to target,
    providing smoother transitions and better target approximation than 
    pure categorical selection.
    
    Args:
        steering_signals: dict, existing steering signals by category
        target_valence: float, target valence value
        target_arousal: float, target arousal value  
        k_neighbors: int, number of nearest neighbors to blend (default: 3)
        distance_power: float, power for distance weighting (default: 2.0)
    
    Returns:
        tuple: (method_name, interpolated_valence_128d, interpolated_arousal_128d)
    """
    import numpy as np
    
    def get_category_center(category_name):
        """Get center coordinates of category."""
        parts = category_name.split('_')
        valence_label, arousal_label = parts[0], parts[1]
        
        valence_centers = {'negative': -0.665, 'neutral': 0.0, 'positive': 0.665}
        arousal_centers = {'weak': -0.665, 'middle': 0.0, 'strong': 0.665}
        
        return valence_centers[valence_label], arousal_centers[arousal_label]
    
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

# Usage example:
# method, val_signal, aro_signal = select_steering_signal_interpolated(
#     steering_signals, target_valence=-0.1, target_arousal=0.3, 
#     k_neighbors=3, distance_power=2.0
# )
'''
    
    with open(f'{output_dir}/interpolation_steering.py', 'w') as f:
        f.write(code)
    
    print(f"🔧 Interpolation implementation saved to {output_dir}/interpolation_steering.py")

def main():
    """Main analysis function."""
    print("🔄 INTERPOLATION-BASED STEERING SIGNAL IMPROVEMENT")
    print("=" * 70)
    
    # Compare selection methods
    steering_signals = compare_selection_methods()
    
    if steering_signals is not None:
        # Test interpolation parameters
        test_interpolation_parameters()
    
    # Analyze benefits
    analyze_interpolation_benefits()
    
    # Generate implementation
    generate_interpolation_implementation()
    
    print("\n✅ INTERPOLATION ANALYSIS COMPLETE!")
    print("\n🎯 **KEY ADVANTAGES:**")
    print("✓ Uses existing 9-bin steering signals (no regeneration needed)")
    print("✓ Provides smooth transitions between categories")
    print("✓ Better target approximation through weighted blending")
    print("✓ Tunable precision via k_neighbors and distance_power")
    print("✓ Reduces boundary sensitivity issues")
    
    print("\n🧪 **RECOMMENDED NEXT STEPS:**")
    print("1. Integrate interpolation into test_steering_signals.py")
    print("2. Test with baseline parameters: k=3, power=2.0")
    print("3. Compare performance vs original categorical method")
    print("4. Test strength variations: 1.0x, 1.2x, 1.5x")
    print("5. Expected: Better performance, especially at boundaries")

if __name__ == '__main__':
    main() 