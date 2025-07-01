#!/usr/bin/env python3
"""
Generate 25-Bin Steering Signals with Smart Fallback

This script generates steering signals using a 25-bin (5×5) categorization system
with intelligent handling of categories that have insufficient samples.
"""

import argparse
import numpy as np
import h5py
import torch
import json
import os
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('src')

from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
from utils.config import cnn6_config

def categorize_25bin_with_fallback(valence, arousal):
    """
    Categorize emotions using 25-bin system with fallback for problematic categories.
    """
    # 5×5 categorization with finer thresholds
    valence_thresholds = [-0.6, -0.2, 0.2, 0.6]
    arousal_thresholds = [-0.6, -0.2, 0.2, 0.6]
    
    valence_labels = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
    arousal_labels = ['very_weak', 'weak', 'middle', 'strong', 'very_strong']
    
    # Categorize valence (5 bins)
    if valence < valence_thresholds[0]:
        v_cat = valence_labels[0]  # very_negative
    elif valence < valence_thresholds[1]:
        v_cat = valence_labels[1]  # negative
    elif valence < valence_thresholds[2]:
        v_cat = valence_labels[2]  # neutral
    elif valence < valence_thresholds[3]:
        v_cat = valence_labels[3]  # positive
    else:
        v_cat = valence_labels[4]  # very_positive
    
    # Categorize arousal (5 bins)
    if arousal < arousal_thresholds[0]:
        a_cat = arousal_labels[0]  # very_weak
    elif arousal < arousal_thresholds[1]:
        a_cat = arousal_labels[1]  # weak
    elif arousal < arousal_thresholds[2]:
        a_cat = arousal_labels[2]  # middle
    elif arousal < arousal_thresholds[3]:
        a_cat = arousal_labels[3]  # strong
    else:
        a_cat = arousal_labels[4]  # very_strong
    
    category_name = f"{v_cat}_{a_cat}"
    
    # Define problematic categories and their fallbacks
    fallback_mapping = {
        'very_positive_very_strong': 'very_positive_strong',  # 1 sample → merge with neighbor
        'negative_very_weak': 'neutral_very_weak',  # 3 samples → merge with neighbor
    }
    
    # Apply fallback if needed
    if category_name in fallback_mapping:
        fallback_category = fallback_mapping[category_name]
        print(f"   Fallback: {category_name} → {fallback_category}")
        return fallback_category
    
    return category_name

def load_and_categorize_data(dataset_path):
    """Load emotion data and categorize using 25-bin system with fallback."""
    print("📂 Loading and categorizing emotion data...")
    
    with h5py.File(dataset_path, 'r') as hf:
        features = hf['feature'][:]
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        audio_names = [name.decode() for name in hf['audio_name'][:]]
    
    print(f"✅ Loaded {len(valence)} samples")
    print(f"   Valence range: [{valence.min():.3f}, {valence.max():.3f}]")
    print(f"   Arousal range: [{arousal.min():.3f}, {arousal.max():.3f}]")
    
    # Categorize all samples
    categories = []
    for v, a in zip(valence, arousal):
        category = categorize_25bin_with_fallback(v, a)
        categories.append(category)
    
    # Count samples per category
    category_counts = Counter(categories)
    
    print(f"\n📊 25-Bin Category Distribution (with fallbacks):")
    for category, count in sorted(category_counts.items()):
        percentage = count / len(valence) * 100
        status = "✅" if count >= 10 else "⚠️" if count >= 5 else "❌"
        print(f"   {category:25}: {count:3d} samples ({percentage:4.1f}%) {status}")
    
    # Check for problematic categories
    problematic = [cat for cat, count in category_counts.items() if count < 5]
    if problematic:
        print(f"\n❌ Still problematic categories: {problematic}")
        print("   Consider additional fallback strategies")
    else:
        print(f"\n✅ All categories have ≥5 samples after fallback")
    
    return features, valence, arousal, audio_names, categories, category_counts

def load_emotion_model(checkpoint_path, device):
    """Load emotion model from checkpoint."""
    print(f"🔧 Loading model from {checkpoint_path}")
    
    # Create model
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
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    return model

def extract_activations_by_category(model, features, categories, category_counts, device, batch_size=8):
    """Extract activations for each category."""
    print("🔍 Extracting activations by category...")
    
    # Group samples by category
    category_samples = defaultdict(list)
    for idx, category in enumerate(categories):
        category_samples[category].append(idx)
    
    # Extract activations for each category
    category_activations = {}
    
    for category, sample_indices in category_samples.items():
        count = len(sample_indices)
        print(f"   Processing {category}: {count} samples")
        
        if count < 5:
            print(f"     ⚠️  Warning: Only {count} samples for {category}")
        
        # Collect activations for this category
        all_activations = {
            'valence_128d': [],
            'arousal_128d': [],
            'valence_256d': [],
            'arousal_256d': [],
            'valence_output': [],
            'arousal_output': [],
            'visual_embedding': []
        }
        
        # Process in batches
        for i in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[i:i + batch_size]
            batch_features = features[batch_indices]
            
            # Convert to tensor and move to device
            batch_tensor = torch.tensor(batch_features, dtype=torch.float32).to(device)
            # LRM model expects 3D input (batch_size, time_steps, mel_bins)
            # No need to add channel dimension - the _forward_visual_system method handles this
            
            with torch.no_grad():
                # Forward pass to extract activations
                output = model(batch_tensor)
                
                # Get visual embedding using the LRM model's method (handles deleted fc layers)
                visual_embedding = model._forward_visual_system(batch_tensor)
                
                # Extract activations from affective pathways (following model structure)
                # Valence pathway: Linear(512,256) -> ReLU -> Linear(256,128) -> ReLU -> Linear(128,1)
                valence_256d = model.affective_valence[0:2](visual_embedding)  # Linear(512,256) + ReLU
                valence_128d = model.affective_valence[2:4](valence_256d)      # Linear(256,128) + ReLU
                valence_output = output['valence']
                
                # Arousal pathway: Linear(512,256) -> ReLU -> Linear(256,128) -> ReLU -> Linear(128,1)
                arousal_256d = model.affective_arousal[0:2](visual_embedding)  # Linear(512,256) + ReLU  
                arousal_128d = model.affective_arousal[2:4](arousal_256d)      # Linear(256,128) + ReLU
                arousal_output = output['arousal']
                
                # Store activations
                all_activations['valence_128d'].append(valence_128d.cpu().numpy())
                all_activations['arousal_128d'].append(arousal_128d.cpu().numpy())
                all_activations['valence_256d'].append(valence_256d.cpu().numpy())
                all_activations['arousal_256d'].append(arousal_256d.cpu().numpy())
                all_activations['valence_output'].append(valence_output.cpu().numpy())
                all_activations['arousal_output'].append(arousal_output.cpu().numpy())
                all_activations['visual_embedding'].append(visual_embedding.cpu().numpy())
        
        # Concatenate and average activations
        category_avg_activations = {}
        for key, activation_list in all_activations.items():
            if activation_list:
                concatenated = np.concatenate(activation_list, axis=0)
                averaged = np.mean(concatenated, axis=0)
                category_avg_activations[key] = averaged
                print(f"     {key}: shape {averaged.shape}, mean {averaged.mean():.4f}")
        
        category_activations[category] = category_avg_activations
    
    return category_activations

def save_25bin_steering_signals(category_activations, output_dir):
    """Save 25-bin steering signals in both directory and JSON formats."""
    print(f"💾 Saving 25-bin steering signals to {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save in directory format (like original)
    for category, activations in category_activations.items():
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        for signal_name, signal_data in activations.items():
            signal_path = os.path.join(category_dir, f"{signal_name}.npy")
            np.save(signal_path, signal_data)
    
    # Save in JSON format (for interpolation testing)
    json_data = {}
    for category, activations in category_activations.items():
        json_data[category] = {}
        for signal_name, signal_data in activations.items():
            json_data[category][signal_name] = signal_data.tolist()
    
    # Add metadata
    json_data['metadata'] = {
        'total_categories': len(category_activations),
        'categorization_method': '25bin_with_fallback',
        'fallback_mapping': {
            'very_positive_very_strong': 'very_positive_strong',
            'negative_very_weak': 'neutral_very_weak'
        }
    }
    
    json_data['generation_config'] = {
        'method': '25bin_with_fallback',
        'categories': len(category_activations),
        'output_dir': output_dir
    }
    
    # Save JSON file
    json_path = os.path.join(output_dir, 'steering_signals_25bin.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✅ Saved {len(category_activations)} categories")
    print(f"   Directory format: {output_dir}/[category]/[signal].npy")
    print(f"   JSON format: {json_path}")
    
    return json_path

def create_25bin_visualization(category_activations, output_dir):
    """Create visualization of 25-bin steering signals."""
    print("📊 Creating 25-bin visualization...")
    
    categories = list(category_activations.keys())
    
    # Extract signal statistics
    valence_means = []
    arousal_means = []
    valence_stds = []
    arousal_stds = []
    
    for category in categories:
        activations = category_activations[category]
        valence_means.append(np.mean(activations['valence_128d']))
        arousal_means.append(np.mean(activations['arousal_128d']))
        valence_stds.append(np.std(activations['valence_128d']))
        arousal_stds.append(np.std(activations['arousal_128d']))
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('25-Bin Steering Signals Analysis', fontsize=16)
    
    # Plot 1: Valence signal means
    ax = axes[0, 0]
    bars = ax.bar(range(len(categories)), valence_means, color='lightblue', edgecolor='navy')
    ax.set_xlabel('Category')
    ax.set_ylabel('Mean Valence Signal')
    ax.set_title('Valence 128D Signal Means by Category')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    
    # Plot 2: Arousal signal means
    ax = axes[0, 1]
    bars = ax.bar(range(len(categories)), arousal_means, color='lightcoral', edgecolor='darkred')
    ax.set_xlabel('Category')
    ax.set_ylabel('Mean Arousal Signal')
    ax.set_title('Arousal 128D Signal Means by Category')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    
    # Plot 3: Signal standard deviations
    ax = axes[1, 0]
    ax.bar(range(len(categories)), valence_stds, alpha=0.7, label='Valence', color='lightblue')
    ax.bar(range(len(categories)), arousal_stds, alpha=0.7, label='Arousal', color='lightcoral')
    ax.set_xlabel('Category')
    ax.set_ylabel('Signal Standard Deviation')
    ax.set_title('Signal Variability by Category')
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    
    # Plot 4: Signal correlation
    ax = axes[1, 1]
    ax.scatter(valence_means, arousal_means, s=100, alpha=0.7)
    for i, category in enumerate(categories):
        ax.annotate(category, (valence_means[i], arousal_means[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Mean Valence Signal')
    ax.set_ylabel('Mean Arousal Signal')
    ax.set_title('Valence vs Arousal Signal Correlation')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/25bin_steering_signals_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 25-bin visualization saved to {output_dir}/25bin_steering_signals_analysis.png")

def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(description='Generate 25-bin steering signals with smart fallback')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to emotion features HDF5 file')
    parser.add_argument('--model_checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='workspaces/25bin_steering_signals', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    
    args = parser.parse_args()
    
    print("🚀 25-BIN STEERING SIGNALS GENERATION")
    print("=" * 60)
    
    # Setup device
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    print(f"🔧 Using device: {device}")
    
    # Load and categorize data
    features, valence, arousal, audio_names, categories, category_counts = load_and_categorize_data(args.dataset_path)
    
    # Load model
    model = load_emotion_model(args.model_checkpoint, device)
    
    # Extract activations by category
    category_activations = extract_activations_by_category(
        model, features, categories, category_counts, device, args.batch_size
    )
    
    # Save steering signals
    json_path = save_25bin_steering_signals(category_activations, args.output_dir)
    
    # Create visualization
    create_25bin_visualization(category_activations, args.output_dir)
    
    print(f"\n✅ **25-BIN STEERING SIGNALS GENERATION COMPLETE!**")
    print(f"   Output directory: {args.output_dir}")
    print(f"   JSON file: {json_path}")
    print(f"   Categories generated: {len(category_activations)}")
    print(f"   Ready for interpolation testing!")

if __name__ == '__main__':
    main() 