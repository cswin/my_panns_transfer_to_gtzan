#!/usr/bin/env python3
"""
Analyze 9-Bin Sample Distribution
=================================

This script analyzes the actual distribution of samples across 9-bin emotion categories
using the real emotion dataset, showing how many samples fall into each category.
"""

import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('src')

def categorize_emotion_9bin(valence, arousal):
    """Categorize emotion into 9-bin system (3x3 grid)."""
    
    # Coarser thresholds for 3x3 grid
    val_thresholds = [-0.3, 0.3]  # Creates 3 bins: negative, neutral, positive
    aro_thresholds = [-0.3, 0.3]  # Creates 3 bins: weak, moderate, strong
    
    # Determine valence category (3 bins)
    if valence <= val_thresholds[0]:
        val_cat = "negative"
    elif valence <= val_thresholds[1]:
        val_cat = "neutral"
    else:
        val_cat = "positive"
    
    # Determine arousal category (3 bins)
    if arousal <= aro_thresholds[0]:
        aro_cat = "weak"
    elif arousal <= aro_thresholds[1]:
        aro_cat = "moderate"
    else:
        aro_cat = "strong"
    
    return f"{val_cat}_{aro_cat}"

def analyze_sample_distribution(dataset_path):
    """Analyze the distribution of samples across 9-bin categories."""
    print(f"🔍 Analyzing sample distribution from: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset file not found: {dataset_path}")
        return None
    
    # Load the dataset
    with h5py.File(dataset_path, 'r') as hf:
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        audio_names = [name.decode() if isinstance(name, bytes) else name for name in hf['audio_name'][:]]
    
    print(f"✅ Loaded {len(valence)} samples")
    print(f"   Valence range: [{valence.min():.3f}, {valence.max():.3f}]")
    print(f"   Arousal range: [{arousal.min():.3f}, {arousal.max():.3f}]")
    
    # Categorize all samples
    categories = []
    for v, a in zip(valence, arousal):
        category = categorize_emotion_9bin(v, a)
        categories.append(category)
    
    # Count samples per category
    category_counts = Counter(categories)
    
    # Define expected categories
    expected_categories = [
        'negative_weak', 'negative_moderate', 'negative_strong',
        'neutral_weak', 'neutral_moderate', 'neutral_strong',
        'positive_weak', 'positive_moderate', 'positive_strong'
    ]
    
    print(f"\n📊 ACTUAL 9-BIN SAMPLE DISTRIBUTION:")
    print("=" * 60)
    
    total_samples = len(valence)
    for category in expected_categories:
        count = category_counts.get(category, 0)
        percentage = count / total_samples * 100
        status = "✅" if count >= 10 else "⚠️" if count >= 5 else "❌"
        print(f"   {category:20}: {count:4d} samples ({percentage:5.1f}%) {status}")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS:")
    print("-" * 40)
    print(f"   Total samples: {total_samples}")
    print(f"   Categories with samples: {len(category_counts)}")
    print(f"   Average samples per category: {total_samples / 9:.1f}")
    print(f"   Min samples in a category: {min(category_counts.values()) if category_counts else 0}")
    print(f"   Max samples in a category: {max(category_counts.values()) if category_counts else 0}")
    
    # Check for problematic categories
    problematic = [cat for cat, count in category_counts.items() if count < 5]
    if problematic:
        print(f"\n❌ PROBLEMATIC CATEGORIES (< 5 samples):")
        for cat in problematic:
            print(f"   - {cat}: {category_counts[cat]} samples")
    else:
        print(f"\n✅ All categories have ≥5 samples")
    
    return category_counts, valence, arousal, categories

def create_distribution_visualization(category_counts, valence, arousal, categories):
    """Create visualization of the actual sample distribution."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define expected categories and colors
    expected_categories = [
        'negative_weak', 'negative_moderate', 'negative_strong',
        'neutral_weak', 'neutral_moderate', 'neutral_strong',
        'positive_weak', 'positive_moderate', 'positive_strong'
    ]
    
    colors = {
        'negative_weak': '#FF6B6B',      # Red
        'negative_moderate': '#FF8E8E',   # Light red
        'negative_strong': '#FFB3B3',     # Very light red
        'neutral_weak': '#4ECDC4',        # Teal
        'neutral_moderate': '#45B7AA',    # Darker teal
        'neutral_strong': '#3CA89B',      # Even darker teal
        'positive_weak': '#95E1D3',       # Light green
        'positive_moderate': '#7DD3C2',   # Green
        'positive_strong': '#65C5B1'      # Dark green
    }
    
    # Plot 1: Bar chart of sample counts
    counts = [category_counts.get(cat, 0) for cat in expected_categories]
    bars = ax1.bar(range(len(expected_categories)), counts, 
                   color=[colors[cat] for cat in expected_categories])
    
    ax1.set_title('Actual Sample Distribution Across 9 Categories', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Emotion Categories', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_xticks(range(len(expected_categories)))
    ax1.set_xticklabels([cat.replace('_', '\n') for cat in expected_categories], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Scatter plot of valence vs arousal with category colors
    ax2.scatter(valence, arousal, c=[colors.get(cat, 'gray') for cat in categories], 
                alpha=0.6, s=20)
    ax2.set_xlabel('Valence', fontsize=12)
    ax2.set_ylabel('Arousal', fontsize=12)
    ax2.set_title('Valence-Arousal Distribution with 9-Bin Categories', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    
    # Add category boundaries
    for v_bound in [-0.3, 0.3]:
        ax2.axvline(v_bound, color='black', linestyle='--', alpha=0.5)
    for a_bound in [-0.3, 0.3]:
        ax2.axhline(a_bound, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Percentage distribution
    total_samples = len(valence)
    percentages = [count / total_samples * 100 for count in counts]
    
    bars = ax3.bar(range(len(expected_categories)), percentages, 
                   color=[colors[cat] for cat in expected_categories])
    ax3.set_title('Percentage Distribution Across 9 Categories', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Emotion Categories', fontsize=12)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_xticks(range(len(expected_categories)))
    ax3.set_xticklabels([cat.replace('_', '\n') for cat in expected_categories], rotation=45, ha='right')
    
    # Add percentage labels
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Category balance analysis
    expected_avg = total_samples / 9
    balance_scores = [(count - expected_avg) / expected_avg * 100 for count in counts]
    
    bars = ax4.bar(range(len(expected_categories)), balance_scores, 
                   color=[colors[cat] for cat in expected_categories])
    ax4.set_title('Category Balance (Deviation from Expected Average)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Emotion Categories', fontsize=12)
    ax4.set_ylabel('Deviation from Average (%)', fontsize=12)
    ax4.set_xticks(range(len(expected_categories)))
    ax4.set_xticklabels([cat.replace('_', '\n') for cat in expected_categories], rotation=45, ha='right')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.grid(True, alpha=0.3)
    
    # Add balance labels
    for bar, balance in zip(bars, balance_scores):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -1),
                f'{balance:+.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/actual_9bin_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return counts, percentages, balance_scores

def main():
    """Main function to analyze sample distribution."""
    print("📊 Analyzing 9-Bin Sample Distribution")
    print("=" * 50)
    
    # Dataset path
    dataset_path = "workspaces/emotoin_feedback/features/emotion_features.h5"
    
    # Analyze distribution
    results = analyze_sample_distribution(dataset_path)
    if results is None:
        return
    
    category_counts, valence, arousal, categories = results
    
    # Create visualization
    print(f"\n🎨 Creating distribution visualization...")
    counts, percentages, balance_scores = create_distribution_visualization(
        category_counts, valence, arousal, categories)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Visualization saved to: visualizations/actual_9bin_distribution.png")
    
    # Print key insights
    print(f"\n🎯 KEY INSIGHTS:")
    print("-" * 30)
    
    # Find most/least populated categories
    most_populated = max(category_counts.items(), key=lambda x: x[1])
    least_populated = min(category_counts.items(), key=lambda x: x[1])
    
    print(f"   Most populated category: {most_populated[0]} ({most_populated[1]} samples)")
    print(f"   Least populated category: {least_populated[0]} ({least_populated[1]} samples)")
    
    # Check balance
    total_samples = len(valence)
    expected_avg = total_samples / 9
    max_deviation = max(abs(score) for score in balance_scores)
    
    print(f"   Expected average per category: {expected_avg:.1f} samples")
    print(f"   Maximum deviation from average: {max_deviation:.1f}%")
    
    if max_deviation > 50:
        print(f"   ⚠️  WARNING: High imbalance detected!")
    elif max_deviation > 25:
        print(f"   ⚠️  Moderate imbalance detected")
    else:
        print(f"   ✅ Good balance across categories")

if __name__ == '__main__':
    main() 