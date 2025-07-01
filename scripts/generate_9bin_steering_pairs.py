#!/usr/bin/env python3
"""
Generate 9 Pairs of Steering Signals for Emotion Feedback (Coarser Binning)

This script uses COARSER emotion categorization (3x3 = 9 bins) to ensure 
sufficient samples per category for reliable steering signal generation.

Emotion Grid:
- Valence: negative, neutral, positive  
- Arousal: weak, moderate, strong
- Total: 9 categories with more samples each
"""

import os
import sys
import argparse
import numpy as np
import torch
import h5py
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Add project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def categorize_emotion_9bin(valence, arousal):
    """Categorize emotion into 9-bin system (3x3 grid) for better sample distribution."""
    
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

def load_emotion_data(dataset_path):
    """Load emotion dataset and categorize into 9 bins."""
    print(f"📁 Loading emotion data from {dataset_path}")
    
    with h5py.File(dataset_path, 'r') as hf:
        features = hf['feature'][:]
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        audio_names = [name.decode() if isinstance(name, bytes) else name for name in hf['audio_name'][:]]
    
    print(f"   Total samples: {len(features)}")
    print(f"   Feature shape: {features[0].shape}")
    
    # Categorize samples using 9-bin system
    categories = []
    for v, a in zip(valence, arousal):
        categories.append(categorize_emotion_9bin(v, a))
    
    # Count samples per category
    category_counts = {}
    for cat in categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"   Found {len(category_counts)} unique categories (9-bin system):")
    for cat, count in sorted(category_counts.items()):
        print(f"     {cat}: {count} samples")
    
    # Check minimum sample requirement
    min_samples = min(category_counts.values()) if category_counts else 0
    print(f"   Minimum samples per category: {min_samples}")
    
    return features, valence, arousal, audio_names, categories, category_counts

def load_model(checkpoint_path, device):
    """Load trained emotion model."""
    print(f"🔧 Loading model from {checkpoint_path}")
    
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, forward_passes=2
    ).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print("   ✅ Model loaded successfully")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    return model

def extract_steering_signal_pairs(model, features, categories, category_counts, device, batch_size=8, min_samples=15):
    """Extract steering signal pairs for categories with sufficient samples."""
    print(f"🧬 Extracting steering signal pairs (min {min_samples} samples per category)...")
    
    # Group samples by category
    category_samples = defaultdict(list)
    for idx, category in enumerate(categories):
        category_samples[category].append(idx)
    
    # Filter categories with insufficient samples
    valid_categories = {cat: indices for cat, indices in category_samples.items() 
                       if len(indices) >= min_samples}
    
    skipped_categories = {cat: len(indices) for cat, indices in category_samples.items() 
                         if len(indices) < min_samples}
    
    if skipped_categories:
        print(f"   ⚠️  Skipping {len(skipped_categories)} categories with <{min_samples} samples:")
        for cat, count in skipped_categories.items():
            print(f"     - {cat}: {count} samples")
    
    print(f"   Processing {len(valid_categories)} categories with sufficient samples")
    
    # Extract activations for each valid category
    steering_pairs = {}
    
    for category, sample_indices in valid_categories.items():
        count = len(sample_indices)
        print(f"   Processing {category}: {count} samples")
        
        # Collect separate activations for valence and arousal pathways
        valence_activations = []
        arousal_activations = []
        
        # Process in batches
        for i in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[i:i + batch_size]
            batch_features = features[batch_indices]
            
            # Convert to tensor and move to device
            batch_tensor = torch.tensor(batch_features, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                # Forward through visual system to get shared embedding
                visual_embedding = model._forward_visual_system(batch_tensor)
                
                # CRITICAL: Extract from SEPARATE pathways
                # Valence pathway: Linear(512,256) -> ReLU -> Linear(256,128) -> ReLU
                valence_256d = model.affective_valence[0:2](visual_embedding)  # First part of valence path
                valence_128d = model.affective_valence[2:4](valence_256d)      # Key layer for valence steering
                
                # Arousal pathway: Linear(512,256) -> ReLU -> Linear(256,128) -> ReLU  
                arousal_256d = model.affective_arousal[0:2](visual_embedding)  # First part of arousal path
                arousal_128d = model.affective_arousal[2:4](arousal_256d)      # Key layer for arousal steering
                
                # Store activations (these should be DIFFERENT!)
                valence_activations.append(valence_128d.cpu().numpy())
                arousal_activations.append(arousal_128d.cpu().numpy())
        
        # Average activations across all samples in this category
        valence_concatenated = np.concatenate(valence_activations, axis=0)
        arousal_concatenated = np.concatenate(arousal_activations, axis=0)
        
        valence_signal = np.mean(valence_concatenated, axis=0)  # [128]
        arousal_signal = np.mean(arousal_concatenated, axis=0)  # [128]
        
        # Validate signals are different
        correlation = pearsonr(valence_signal.flatten(), arousal_signal.flatten())[0]
        mean_diff = np.mean(np.abs(valence_signal - arousal_signal))
        
        print(f"     Valence signal: mean={valence_signal.mean():.4f}, std={valence_signal.std():.4f}")
        print(f"     Arousal signal: mean={arousal_signal.mean():.4f}, std={arousal_signal.std():.4f}")
        print(f"     Correlation: {correlation:.4f}, Mean diff: {mean_diff:.6f}")
        
        if correlation > 0.95:
            print(f"     ⚠️  WARNING: Valence and arousal signals are very similar!")
        elif mean_diff < 0.001:
            print(f"     ⚠️  WARNING: Signals are nearly identical!")
        else:
            print(f"     ✅ Signals are properly differentiated")
        
        # Store the pair
        steering_pairs[category] = {
            'valence_128d': valence_signal,
            'arousal_128d': arousal_signal,
            'samples_count': count,
            'correlation': correlation,
            'mean_difference': mean_diff
        }
    
    print(f"✅ Extracted {len(steering_pairs)} steering signal pairs (9-bin system)")
    return steering_pairs

def validate_signal_pairs(steering_pairs):
    """Validate that the extracted signal pairs are properly differentiated."""
    print(f"\n🔍 SIGNAL PAIR VALIDATION (9-BIN)")
    print(f"{'Category':<20} {'Samples':<8} {'Correlation':<12} {'Mean Diff':<12} {'Status':<10}")
    print("-" * 75)
    
    identical_count = 0
    high_correlation_count = 0
    good_pairs_count = 0
    
    for category, signals in steering_pairs.items():
        correlation = signals['correlation']
        mean_diff = signals['mean_difference']
        samples = signals['samples_count']
        
        if correlation > 0.98 or mean_diff < 0.0001:
            status = "IDENTICAL"
            identical_count += 1
        elif correlation > 0.9:
            status = "TOO_SIMILAR"
            high_correlation_count += 1
        else:
            status = "GOOD"
            good_pairs_count += 1
        
        print(f"{category:<20} {samples:<8} {correlation:<12.4f} {mean_diff:<12.6f} {status:<10}")
    
    total_pairs = len(steering_pairs)
    print(f"\n📊 VALIDATION SUMMARY (9-BIN):")
    print(f"   Total pairs: {total_pairs}")
    print(f"   Good pairs: {good_pairs_count} ({good_pairs_count/total_pairs*100:.1f}%)")
    print(f"   Too similar: {high_correlation_count} ({high_correlation_count/total_pairs*100:.1f}%)")
    print(f"   Identical: {identical_count} ({identical_count/total_pairs*100:.1f}%)")
    
    if identical_count > 0:
        print(f"   ❌ CRITICAL: {identical_count} pairs are identical!")
        return False
    elif high_correlation_count > total_pairs * 0.5:
        print(f"   ⚠️  WARNING: Too many similar pairs")
        return False
    else:
        print(f"   ✅ Signal pairs are properly differentiated")
        return True

def save_steering_pairs(steering_pairs, output_dir):
    """Save the steering signal pairs."""
    print(f"💾 Saving steering signal pairs to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for JSON format
    json_data = {}
    
    for category, signals in steering_pairs.items():
        json_data[category] = {
            'valence_128d': signals['valence_128d'].tolist(),
            'arousal_128d': signals['arousal_128d'].tolist(),
            'samples_count': signals['samples_count'],
            'correlation': float(signals['correlation']),
            'mean_difference': float(signals['mean_difference'])
        }
    
    # Add metadata
    json_data['metadata'] = {
        'total_pairs': len(steering_pairs),
        'extraction_method': 'separate_pathway_averaging_9bin',
        'binning_system': '9bin_coarse',
        'signal_dimensions': {
            'valence_128d': 128,
            'arousal_128d': 128
        }
    }
    
    json_data['generation_config'] = {
        'method': '9bin_pairs',
        'categories': len(steering_pairs),
        'validation_passed': validate_signal_pairs(steering_pairs),
        'output_dir': output_dir
    }
    
    # Save JSON file
    json_path = os.path.join(output_dir, 'steering_signal_pairs_9bin.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    # Also save individual .npy files for compatibility
    for category, signals in steering_pairs.items():
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        np.save(os.path.join(category_dir, 'valence_128d.npy'), signals['valence_128d'])
        np.save(os.path.join(category_dir, 'arousal_128d.npy'), signals['arousal_128d'])
    
    print(f"✅ Saved {len(steering_pairs)} steering signal pairs (9-bin)")
    print(f"   JSON format: {json_path}")
    print(f"   Directory format: {output_dir}/[category]/[valence|arousal]_128d.npy")
    
    return json_path

def create_9bin_analysis_plots(steering_pairs, output_dir):
    """Create analysis plots for the 9-bin steering signal pairs."""
    print(f"📊 Creating 9-bin pair analysis plots...")
    
    categories = list(steering_pairs.keys())
    
    # Extract statistics
    valence_means = [np.mean(signals['valence_128d']) for category, signals in steering_pairs.items()]
    arousal_means = [np.mean(signals['arousal_128d']) for category, signals in steering_pairs.items()]
    correlations = [signals['correlation'] for category, signals in steering_pairs.items()]
    sample_counts = [signals['samples_count'] for category, signals in steering_pairs.items()]
    
    # Create comprehensive analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('9-Bin Steering Signal Pairs Analysis', fontsize=16)
    
    # Plot 1: Valence vs Arousal signal means (colored by sample count)
    ax = axes[0, 0]
    scatter = ax.scatter(valence_means, arousal_means, c=sample_counts, s=150, alpha=0.8, cmap='viridis')
    ax.set_xlabel('Mean Valence Signal')
    ax.set_ylabel('Mean Arousal Signal')
    ax.set_title('Valence vs Arousal Signal Means')
    plt.colorbar(scatter, ax=ax, label='Sample Count')
    ax.grid(True, alpha=0.3)
    
    # Annotate points with category names
    for i, category in enumerate(categories):
        ax.annotate(category, (valence_means[i], arousal_means[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    # Plot 2: Sample counts per category
    ax = axes[0, 1]
    bars = ax.bar(range(len(categories)), sample_counts, alpha=0.7, color='skyblue', edgecolor='navy')
    ax.set_xlabel('Category')
    ax.set_ylabel('Sample Count')
    ax.set_title('Samples per Category (9-bin)')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    
    # Add count labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom')
    
    # Plot 3: Signal correlations
    ax = axes[1, 0]
    bars = ax.bar(range(len(categories)), correlations, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax.set_xlabel('Category')
    ax.set_ylabel('Valence-Arousal Correlation')
    ax.set_title('Signal Correlations by Category')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.axhline(0.9, color='red', linestyle='--', label='High Correlation Threshold')
    ax.legend()
    
    # Plot 4: Emotion space coverage
    ax = axes[1, 1]
    
    # Create emotion grid visualization
    val_positions = {'negative': 0, 'neutral': 1, 'positive': 2}
    aro_positions = {'weak': 0, 'moderate': 1, 'strong': 2}
    
    # Create grid
    grid = np.zeros((3, 3))
    grid_labels = np.empty((3, 3), dtype=object)
    
    for category in categories:
        val_cat, aro_cat = category.split('_')
        val_pos = val_positions[val_cat]
        aro_pos = aro_positions[aro_cat]
        grid[aro_pos, val_pos] = steering_pairs[category]['samples_count']
        grid_labels[aro_pos, val_pos] = f"{category}\n({steering_pairs[category]['samples_count']})"
    
    im = ax.imshow(grid, cmap='Blues', aspect='auto')
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_title('9-Bin Emotion Space Coverage')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Weak', 'Moderate', 'Strong'])
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            if grid_labels[i, j] is not None:
                ax.text(j, i, grid_labels[i, j], ha="center", va="center", fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Sample Count')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'steering_pairs_9bin_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Analysis plots saved to {plot_path}")

def main():
    """Main function to generate 9 pairs of steering signals."""
    parser = argparse.ArgumentParser(description='Generate 9 pairs of steering signals (coarse 3x3 binning)')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to emotion features HDF5 file')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./steering_signal_pairs_9bin', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--min_samples', type=int, default=15, help='Minimum samples per category')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    
    args = parser.parse_args()
    
    print("🚀 9-BIN STEERING SIGNAL PAIRS GENERATION")
    print("=" * 60)
    
    # Setup device
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Load data
    features, valence, arousal, audio_names, categories, category_counts = load_emotion_data(args.dataset_path)
    
    # Load model
    model = load_model(args.model_checkpoint, device)
    
    # Extract steering signal pairs
    steering_pairs = extract_steering_signal_pairs(
        model, features, categories, category_counts, device, args.batch_size, args.min_samples
    )
    
    # Validate pairs
    validation_passed = validate_signal_pairs(steering_pairs)
    
    if not validation_passed:
        print("⚠️  WARNING: Validation failed - signals may not be properly differentiated!")
        print("   Consider using different extraction methods or model layers.")
    
    # Save results
    json_path = save_steering_pairs(steering_pairs, args.output_dir)
    
    # Create analysis plots
    create_9bin_analysis_plots(steering_pairs, args.output_dir)
    
    print(f"\n🎉 GENERATION COMPLETE!")
    print(f"   Generated {len(steering_pairs)} steering signal pairs (9-bin system)")
    print(f"   Total signals: {len(steering_pairs) * 2}")
    print(f"   Output: {args.output_dir}")
    print(f"   JSON file: {json_path}")
    print(f"   Validation: {'✅ PASSED' if validation_passed else '❌ FAILED'}")

if __name__ == "__main__":
    main() 