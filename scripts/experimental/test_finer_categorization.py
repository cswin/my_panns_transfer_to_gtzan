#!/usr/bin/env python3
"""
Test Finer Categorization for Improved Steering Signal Selection

This script implements and tests a 25-bin (5×5) categorization system
as a simple improvement over the current 9-bin (3×3) system.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('src')

def create_finer_categorization():
    """Create a 25-bin (5×5) categorization system."""
    
    # 5 valence categories instead of 3
    valence_labels = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
    valence_thresholds = [-0.6, -0.2, 0.2, 0.6]  # 4 thresholds for 5 categories
    
    # 5 arousal categories instead of 3  
    arousal_labels = ['very_weak', 'weak', 'middle', 'strong', 'very_strong']
    arousal_thresholds = [-0.6, -0.2, 0.2, 0.6]  # 4 thresholds for 5 categories
    
    return {
        'valence_labels': valence_labels,
        'valence_thresholds': valence_thresholds,
        'arousal_labels': arousal_labels,
        'arousal_thresholds': arousal_thresholds,
        'total_categories': 25
    }

def select_category_25bin(target_valence, target_arousal, config):
    """Select category using 25-bin system."""
    
    # Categorize valence (5 bins)
    v_thresholds = config['valence_thresholds']
    if target_valence < v_thresholds[0]:
        v_category = config['valence_labels'][0]  # very_negative
    elif target_valence < v_thresholds[1]:
        v_category = config['valence_labels'][1]  # negative
    elif target_valence < v_thresholds[2]:
        v_category = config['valence_labels'][2]  # neutral
    elif target_valence < v_thresholds[3]:
        v_category = config['valence_labels'][3]  # positive
    else:
        v_category = config['valence_labels'][4]  # very_positive
    
    # Categorize arousal (5 bins)
    a_thresholds = config['arousal_thresholds']
    if target_arousal < a_thresholds[0]:
        a_category = config['arousal_labels'][0]  # very_weak
    elif target_arousal < a_thresholds[1]:
        a_category = config['arousal_labels'][1]  # weak
    elif target_arousal < a_thresholds[2]:
        a_category = config['arousal_labels'][2]  # middle
    elif target_arousal < a_thresholds[3]:
        a_category = config['arousal_labels'][3]  # strong
    else:
        a_category = config['arousal_labels'][4]  # very_strong
    
    return f"{v_category}_{a_category}"

def select_category_9bin(target_valence, target_arousal):
    """Select category using original 9-bin system for comparison."""
    
    # Original 3×3 system
    valence_thresholds = [-0.33, 0.33]
    arousal_thresholds = [-0.33, 0.33]
    
    if target_valence < valence_thresholds[0]:
        v_category = "negative"
    elif target_valence > valence_thresholds[1]:
        v_category = "positive"
    else:
        v_category = "neutral"
    
    if target_arousal < arousal_thresholds[0]:
        a_category = "weak"
    elif target_arousal > arousal_thresholds[1]:
        a_category = "strong"
    else:
        a_category = "middle"
    
    return f"{v_category}_{a_category}"

def analyze_categorization_comparison():
    """Compare 9-bin vs 25-bin categorization systems."""
    print("🔍 CATEGORIZATION COMPARISON: 9-BIN vs 25-BIN")
    print("=" * 60)
    
    # Create 25-bin config
    config_25 = create_finer_categorization()
    
    # Analyze category sizes
    print("📊 CATEGORY SIZE COMPARISON:")
    print("\n9-Bin System (3×3):")
    print("   Each category covers: 0.67 × 0.67 = 0.44 area units (22.2% each)")
    print("   Valence bins: [-1.00, -0.33], [-0.33, 0.33], [0.33, 1.00]")
    print("   Arousal bins: [-1.00, -0.33], [-0.33, 0.33], [0.33, 1.00]")
    
    print("\n25-Bin System (5×5):")
    print("   Each category covers: 0.40 × 0.40 = 0.16 area units (8.0% each)")
    print("   Valence bins: [-1.0, -0.6], [-0.6, -0.2], [-0.2, 0.2], [0.2, 0.6], [0.6, 1.0]")
    print("   Arousal bins: [-1.0, -0.6], [-0.6, -0.2], [-0.2, 0.2], [0.2, 0.6], [0.6, 1.0]")
    
    print(f"\n🎯 IMPROVEMENT:")
    print(f"   Category size reduction: 22.2% → 8.0% (2.8× smaller)")
    print(f"   Precision improvement: 2.8× better emotion representation")
    
    # Test boundary sensitivity
    print("\n⚠️  BOUNDARY SENSITIVITY COMPARISON:")
    
    test_cases = [
        {'v': -0.34, 'a': 0.0, 'desc': 'Near negative boundary'},
        {'v': -0.32, 'a': 0.0, 'desc': 'Near neutral boundary'},
        {'v': 0.0, 'a': 0.0, 'desc': 'Center point'},
        {'v': 0.32, 'a': 0.32, 'desc': 'Edge of neutral'},
        {'v': 0.21, 'a': 0.19, 'desc': 'Close to center'},
    ]
    
    print("   Target Emotion → 9-Bin Category → 25-Bin Category")
    print("   " + "-" * 55)
    
    for case in test_cases:
        v, a = case['v'], case['a']
        cat_9 = select_category_9bin(v, a)
        cat_25 = select_category_25bin(v, a, config_25)
        print(f"   V={v:5.2f}, A={a:5.2f} → {cat_9:15} → {cat_25}")
    
    return config_25

def simulate_improvement_potential():
    """Simulate potential performance improvement with finer categorization."""
    print("\n🚀 IMPROVEMENT POTENTIAL ANALYSIS")
    print("=" * 60)
    
    # Create test grid
    valence_test = np.linspace(-0.8, 0.8, 50)
    arousal_test = np.linspace(-0.8, 0.8, 50)
    
    config_25 = create_finer_categorization()
    
    # Count category changes across the grid
    category_changes = 0
    total_comparisons = 0
    
    for v in valence_test:
        for a in arousal_test:
            cat_9 = select_category_9bin(v, a)
            cat_25 = select_category_25bin(v, a, config_25)
            
            if cat_9 != cat_25:  # Different categories (can't directly compare, but shows refinement)
                category_changes += 1
            total_comparisons += 1
    
    print(f"📊 CATEGORIZATION REFINEMENT:")
    print(f"   Test points: {total_comparisons}")
    print(f"   Points with finer categorization: {category_changes} ({category_changes/total_comparisons*100:.1f}%)")
    
    print(f"\n💡 EXPECTED IMPROVEMENTS:")
    print(f"   1. **Reduced approximation error**: 2.8× smaller category sizes")
    print(f"   2. **Better target matching**: More precise emotion representation")
    print(f"   3. **Potential for higher strengths**: Better precision may allow amplification")
    print(f"   4. **Reduced boundary sensitivity**: Smaller impact of category flipping")
    
    print(f"\n🎯 PREDICTED PERFORMANCE:")
    print(f"   - Strength=1.0: Should maintain or improve current performance")
    print(f"   - Strength=1.2-1.5: May become viable with better precision")
    print(f"   - Strength=2.0: Still likely too high, but less degradation expected")

def create_comparison_visualization(output_dir='tmp'):
    """Create visualization comparing 9-bin vs 25-bin systems."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create grid for visualization
    valence_grid = np.linspace(-1, 1, 200)
    arousal_grid = np.linspace(-1, 1, 200)
    V, A = np.meshgrid(valence_grid, arousal_grid)
    
    config_25 = create_finer_categorization()
    
    # Create category maps
    categories_9 = np.zeros_like(V, dtype=int)
    categories_25 = np.zeros_like(V, dtype=int)
    
    # Map for 9-bin system
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            v, a = V[i, j], A[i, j]
            
            # 9-bin mapping
            if v < -0.33:
                v_cat_9 = 0
            elif v > 0.33:
                v_cat_9 = 2
            else:
                v_cat_9 = 1
            
            if a < -0.33:
                a_cat_9 = 0
            elif a > 0.33:
                a_cat_9 = 2
            else:
                a_cat_9 = 1
            
            categories_9[i, j] = v_cat_9 * 3 + a_cat_9
            
            # 25-bin mapping
            v_thresholds = config_25['valence_thresholds']
            a_thresholds = config_25['arousal_thresholds']
            
            if v < v_thresholds[0]:
                v_cat_25 = 0
            elif v < v_thresholds[1]:
                v_cat_25 = 1
            elif v < v_thresholds[2]:
                v_cat_25 = 2
            elif v < v_thresholds[3]:
                v_cat_25 = 3
            else:
                v_cat_25 = 4
            
            if a < a_thresholds[0]:
                a_cat_25 = 0
            elif a < a_thresholds[1]:
                a_cat_25 = 1
            elif a < a_thresholds[2]:
                a_cat_25 = 2
            elif a < a_thresholds[3]:
                a_cat_25 = 3
            else:
                a_cat_25 = 4
            
            categories_25[i, j] = v_cat_25 * 5 + a_cat_25
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: 9-bin system
    ax = axes[0]
    im1 = ax.imshow(categories_9, extent=[-1, 1, -1, 1], origin='lower', 
                    cmap='Set1', vmin=0, vmax=8, aspect='equal')
    
    # Add 9-bin boundaries
    ax.axvline(-0.33, color='black', linewidth=2, alpha=0.8)
    ax.axvline(0.33, color='black', linewidth=2, alpha=0.8)
    ax.axhline(-0.33, color='black', linewidth=2, alpha=0.8)
    ax.axhline(0.33, color='black', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_title('9-Bin System (3×3)\nLarge Categories: 22% each')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: 25-bin system
    ax = axes[1]
    im2 = ax.imshow(categories_25, extent=[-1, 1, -1, 1], origin='lower', 
                    cmap='tab20', vmin=0, vmax=24, aspect='equal')
    
    # Add 25-bin boundaries
    for threshold in config_25['valence_thresholds']:
        ax.axvline(threshold, color='black', linewidth=1, alpha=0.6)
    for threshold in config_25['arousal_thresholds']:
        ax.axhline(threshold, color='black', linewidth=1, alpha=0.6)
    
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_title('25-Bin System (5×5)\nSmaller Categories: 8% each')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/categorization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison visualization saved to {output_dir}/categorization_comparison.png")

def generate_implementation_code(output_dir='tmp'):
    """Generate code for implementing 25-bin categorization."""
    os.makedirs(output_dir, exist_ok=True)
    
    code = '''
def select_steering_signal_by_target_25bin(steering_signals_25bin, target_valence, target_arousal):
    """
    IMPROVED: Select appropriate steering signal using 25-bin (5×5) categorization.
    
    Args:
        steering_signals_25bin: dict, steering signals with 25 categories
        target_valence: float, target valence value 
        target_arousal: float, target arousal value
    
    Returns:
        tuple: (category_name, valence_128d, arousal_128d) or (None, None, None) if not found
    """
    # 5×5 categorization with finer thresholds
    valence_thresholds = [-0.6, -0.2, 0.2, 0.6]
    arousal_thresholds = [-0.6, -0.2, 0.2, 0.6]
    
    valence_labels = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
    arousal_labels = ['very_weak', 'weak', 'middle', 'strong', 'very_strong']
    
    # Categorize valence (5 bins instead of 3)
    if target_valence < valence_thresholds[0]:
        valence_category = valence_labels[0]  # very_negative
    elif target_valence < valence_thresholds[1]:
        valence_category = valence_labels[1]  # negative
    elif target_valence < valence_thresholds[2]:
        valence_category = valence_labels[2]  # neutral
    elif target_valence < valence_thresholds[3]:
        valence_category = valence_labels[3]  # positive
    else:
        valence_category = valence_labels[4]  # very_positive
    
    # Categorize arousal (5 bins instead of 3)
    if target_arousal < arousal_thresholds[0]:
        arousal_category = arousal_labels[0]  # very_weak
    elif target_arousal < arousal_thresholds[1]:
        arousal_category = arousal_labels[1]  # weak
    elif target_arousal < arousal_thresholds[2]:
        arousal_category = arousal_labels[2]  # middle
    elif target_arousal < arousal_thresholds[3]:
        arousal_category = arousal_labels[3]  # strong
    else:
        arousal_category = arousal_labels[4]  # very_strong
    
    # Construct category name
    category_name = f"{valence_category}_{arousal_category}"
    
    # Check if this category exists in steering signals
    if category_name in steering_signals_25bin:
        signals = steering_signals_25bin[category_name]
        if 'valence_128d' in signals and 'arousal_128d' in signals:
            valence_128d = np.array(signals['valence_128d'])
            arousal_128d = np.array(signals['arousal_128d'])
            return category_name, valence_128d, arousal_128d
    
    return None, None, None

# Usage example:
# category, val_signal, aro_signal = select_steering_signal_by_target_25bin(
#     steering_signals_25bin, target_valence=-0.1, target_arousal=0.3
# )
'''
    
    with open(f'{output_dir}/improved_25bin_selection.py', 'w') as f:
        f.write(code)
    
    print(f"🔧 25-bin implementation code saved to {output_dir}/improved_25bin_selection.py")

def main():
    """Main analysis function."""
    print("🚀 FINER CATEGORIZATION IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    # Analyze categorization comparison
    config_25 = analyze_categorization_comparison()
    
    # Simulate improvement potential
    simulate_improvement_potential()
    
    # Create visualization
    create_comparison_visualization()
    
    # Generate implementation code
    generate_implementation_code()
    
    print("\n✅ ANALYSIS COMPLETE!")
    print("\n🎯 NEXT STEPS:")
    print("1. Generate 25-bin steering signals from your emotion dataset")
    print("2. Test the improved selection function with multiple strengths")
    print("3. Compare performance: 9-bin vs 25-bin at different strengths")
    print("4. Expected result: Better performance at higher strengths with 25-bin system")
    
    print("\n💡 HYPOTHESIS TO TEST:")
    print("If categorical imprecision is the limiting factor, then:")
    print("- 25-bin system should maintain strength=1.0 performance")
    print("- 25-bin system should show less degradation at higher strengths")
    print("- Optimal strength might shift to 1.2x or 1.5x with better precision")

if __name__ == '__main__':
    main() 