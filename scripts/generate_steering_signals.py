#!/usr/bin/env python3
"""
Generate External Steering Signals for Emotion Feedback Testing

This script creates external steering signals by:
1. Loading emotion dataset and categorizing audio files into 9 bins based on valence/arousal
2. Extracting average activations from each category
3. Saving steering signals for testing feedback mechanisms

The 9 bins are:
- Valence: Positive, Neutral, Negative (3 levels)
- Arousal: Strong, Middle, Weak (3 levels)
- Total: 3 x 3 = 9 categories

Usage:
    python scripts/generate_steering_signals.py --dataset_path features/emotion_features.h5 --output_dir steering_signals
"""

import os
import argparse
import numpy as np
import torch
import h5py
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Add src to path for imports
import sys
sys.path.append('src')

from data.data_generator import EmoSoundscapesDataset, EmotionValidateSampler, emotion_collate_fn
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
from utils.config import cnn6_config

def categorize_emotion_data(valence, arousal, method='quantile'):
    """
    Categorize emotion data into 9 bins based on valence and arousal.
    
    Args:
        valence: array of valence values
        arousal: array of arousal values  
        method: 'quantile' or 'fixed_threshold'
    
    Returns:
        categories: array of category labels (0-8)
        category_info: dict with category boundaries and descriptions
    """
    if method == 'quantile':
        # Use quantiles to split into 3 equal groups
        valence_33, valence_67 = np.percentile(valence, [33.33, 66.67])
        arousal_33, arousal_67 = np.percentile(arousal, [33.33, 66.67])
        
        category_info = {
            'valence_bounds': {
                'negative': (-np.inf, valence_33),
                'neutral': (valence_33, valence_67), 
                'positive': (valence_67, np.inf)
            },
            'arousal_bounds': {
                'weak': (-np.inf, arousal_33),
                'middle': (arousal_33, arousal_67),
                'strong': (arousal_67, np.inf)
            }
        }
        
    elif method == 'fixed_threshold':
        # Use fixed thresholds based on typical emotion ranges
        category_info = {
            'valence_bounds': {
                'negative': (-np.inf, 0.3),
                'neutral': (0.3, 0.7),
                'positive': (0.7, np.inf)
            },
            'arousal_bounds': {
                'weak': (-np.inf, 0.3),
                'middle': (0.3, 0.7), 
                'strong': (0.7, np.inf)
            }
        }
    
    # Create category mapping
    valence_categories = ['negative', 'neutral', 'positive']
    arousal_categories = ['weak', 'middle', 'strong']
    
    categories = []
    for i, (v, a) in enumerate(zip(valence, arousal)):
        # Determine valence category
        v_cat = None
        for cat, (low, high) in category_info['valence_bounds'].items():
            if low <= v < high:
                v_cat = cat
                break
        
        # Determine arousal category  
        a_cat = None
        for cat, (low, high) in category_info['arousal_bounds'].items():
            if low <= a < high:
                a_cat = cat
                break
        
        # Map to 0-8 category index
        if v_cat and a_cat:
            v_idx = valence_categories.index(v_cat)
            a_idx = arousal_categories.index(a_cat)
            category_idx = v_idx * 3 + a_idx
        else:
            category_idx = 4  # Default to middle category
            
        categories.append(category_idx)
    
    # Add category descriptions
    category_descriptions = {}
    for v_idx, v_cat in enumerate(valence_categories):
        for a_idx, a_cat in enumerate(arousal_categories):
            cat_idx = v_idx * 3 + a_idx
            category_descriptions[cat_idx] = f"{v_cat}_{a_cat}"
    
    category_info['descriptions'] = category_descriptions
    
    return np.array(categories), category_info


def extract_activations_from_model(model, data_loader, device):
    """
    Extract activations from the model for all samples in the dataset.
    
    Args:
        model: trained emotion model
        data_loader: data loader for the dataset
        device: torch device
    
    Returns:
        activations: dict with different activation types
        audio_names: list of audio names
    """
    model.eval()
    
    # Storage for activations
    activations = {
        'valence_128d': [],
        'arousal_128d': [],
        'valence_256d': [],
        'arousal_256d': [],
        'visual_embedding': [],
        'valence_output': [],
        'arousal_output': []
    }
    
    audio_names = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            # Move data to device
            features = batch_data['feature'].to(device)
            if len(features.shape) == 3:
                features = features.unsqueeze(1)  # Add channel dimension
            
            # Forward pass to get activations
            # We need to modify the forward pass to return intermediate activations
            batch_activations = model.extract_activations(features)
            
            # Store activations
            for key in activations:
                if key in batch_activations:
                    activations[key].append(batch_activations[key].cpu().numpy())
            
            # Store audio names
            audio_names.extend(batch_data['audio_name'])
    
    # Concatenate all batches
    for key in activations:
        if activations[key]:
            activations[key] = np.concatenate(activations[key], axis=0)
    
    return activations, audio_names


def compute_category_steering_signals(activations, categories, category_info):
    """
    Compute average activations for each category to create steering signals.
    
    Args:
        activations: dict of activation arrays
        categories: array of category labels
        category_info: dict with category information
    
    Returns:
        steering_signals: dict with steering signals for each category
    """
    steering_signals = {}
    
    # Get unique categories
    unique_categories = np.unique(categories)
    
    for cat_idx in unique_categories:
        # Get indices for this category
        cat_indices = np.where(categories == cat_idx)[0]
        
        if len(cat_indices) == 0:
            continue
            
        cat_name = category_info['descriptions'][cat_idx]
        steering_signals[cat_name] = {}
        
        # Compute average activations for this category
        for activation_type, activation_data in activations.items():
            if len(activation_data) > 0:
                cat_activations = activation_data[cat_indices]
                avg_activation = np.mean(cat_activations, axis=0)
                steering_signals[cat_name][activation_type] = avg_activation
    
    return steering_signals


def save_steering_signals(steering_signals, category_info, output_dir):
    """
    Save steering signals and category information to files.
    
    Args:
        steering_signals: dict with steering signals
        category_info: dict with category information
        output_dir: directory to save files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save steering signals as numpy arrays
    for cat_name, signals in steering_signals.items():
        cat_dir = os.path.join(output_dir, cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        
        for signal_type, signal_data in signals.items():
            signal_path = os.path.join(cat_dir, f"{signal_type}.npy")
            np.save(signal_path, signal_data)
    
    # Save category information
    category_info_path = os.path.join(output_dir, "category_info.json")
    with open(category_info_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_info = {}
        for key, value in category_info.items():
            if isinstance(value, dict):
                json_info[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, tuple):
                        json_info[key][subkey] = [float(x) if x != -np.inf else -float('inf') 
                                                for x in subvalue]
                    else:
                        json_info[key][subkey] = subvalue
            else:
                json_info[key] = value
        json.dump(json_info, f, indent=2)
    
    # Save summary statistics
    summary_path = os.path.join(output_dir, "steering_signals_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Steering Signals Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Category Information:\n")
        for cat_idx, desc in category_info['descriptions'].items():
            f.write(f"  Category {cat_idx}: {desc}\n")
        
        f.write(f"\nValence Boundaries:\n")
        for cat, (low, high) in category_info['valence_bounds'].items():
            f.write(f"  {cat}: [{low:.3f}, {high:.3f})\n")
        
        f.write(f"\nArousal Boundaries:\n")
        for cat, (low, high) in category_info['arousal_bounds'].items():
            f.write(f"  {cat}: [{low:.3f}, {high:.3f})\n")
        
        f.write(f"\nSteering Signals Generated:\n")
        for cat_name, signals in steering_signals.items():
            f.write(f"  {cat_name}:\n")
            for signal_type, signal_data in signals.items():
                f.write(f"    {signal_type}: shape {signal_data.shape}\n")

    # Save all steering signals as a JSON file for compatibility with test_25bin_comprehensive.py
    json_data = {}
    for cat_name, signals in steering_signals.items():
        json_data[cat_name] = {}
        for signal_type, signal_data in signals.items():
            json_data[cat_name][signal_type] = signal_data.tolist()
    # Add metadata for compatibility
    json_data['metadata'] = {
        'total_categories': len(steering_signals),
        'categorization_method': '9bin',
    }
    json_data['generation_config'] = {
        'method': '9bin',
        'categories': len(steering_signals),
        'output_dir': output_dir
    }
    json_path = os.path.join(output_dir, 'steering_signals_by_category.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✅ Saved 9-bin steering signals JSON: {json_path}")


def visualize_categories(valence, arousal, categories, category_info, output_dir):
    """
    Create visualization of the emotion categories.
    
    Args:
        valence: array of valence values
        arousal: array of arousal values
        categories: array of category labels
        category_info: dict with category information
        output_dir: directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Define colors for each category
    colors = plt.cm.Set3(np.linspace(0, 1, 9))
    
    # Plot points colored by category
    for cat_idx in range(9):
        if cat_idx in category_info['descriptions']:
            mask = categories == cat_idx
            if np.any(mask):
                plt.scatter(valence[mask], arousal[mask], 
                          c=[colors[cat_idx]], label=category_info['descriptions'][cat_idx],
                          alpha=0.7, s=30)
    
    # Add category boundaries
    valence_bounds = category_info['valence_bounds']
    arousal_bounds = category_info['arousal_bounds']
    
    # Vertical lines for valence boundaries
    for cat, (low, high) in valence_bounds.items():
        if low != -np.inf:
            plt.axvline(x=low, color='gray', linestyle='--', alpha=0.5)
        if high != np.inf:
            plt.axvline(x=high, color='gray', linestyle='--', alpha=0.5)
    
    # Horizontal lines for arousal boundaries
    for cat, (low, high) in arousal_bounds.items():
        if low != -np.inf:
            plt.axhline(y=low, color='gray', linestyle='--', alpha=0.5)
        if high != np.inf:
            plt.axhline(y=high, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.title('Emotion Categories (9 Bins)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, "emotion_categories.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create category distribution bar plot
    plt.figure(figsize=(12, 6))
    
    category_counts = {}
    for cat_idx in range(9):
        if cat_idx in category_info['descriptions']:
            count = np.sum(categories == cat_idx)
            category_counts[category_info['descriptions'][cat_idx]] = count
    
    categories_list = list(category_counts.keys())
    counts_list = list(category_counts.values())
    
    bars = plt.bar(range(len(categories_list)), counts_list, color=colors[:len(categories_list)])
    plt.xlabel('Emotion Category')
    plt.ylabel('Number of Audio Files')
    plt.title('Distribution of Audio Files Across Emotion Categories')
    plt.xticks(range(len(categories_list)), categories_list, rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts_list):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(counts_list),
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    dist_plot_path = os.path.join(output_dir, "category_distribution.png")
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate external steering signals for emotion feedback testing')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to emotion_features.h5 dataset file')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='steering_signals',
                       help='Output directory for steering signals')
    parser.add_argument('--categorization_method', type=str, default='quantile',
                       choices=['quantile', 'fixed_threshold'],
                       help='Method for categorizing emotions')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for processing')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA if available')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    
    args = parser.parse_args()
    
    # Set device
    if args.cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    with h5py.File(args.dataset_path, 'r') as hf:
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        audio_names = hf['audio_name'][:]
    
    print(f"Dataset loaded: {len(valence)} audio files")
    print(f"Valence range: [{valence.min():.3f}, {valence.max():.3f}]")
    print(f"Arousal range: [{arousal.min():.3f}, {arousal.max():.3f}]")
    
    # Categorize emotion data
    print(f"Categorizing emotions using {args.categorization_method} method...")
    categories, category_info = categorize_emotion_data(valence, arousal, args.categorization_method)
    
    # Print category distribution
    print("\nCategory distribution:")
    for cat_idx in range(9):
        if cat_idx in category_info['descriptions']:
            count = np.sum(categories == cat_idx)
            print(f"  {category_info['descriptions'][cat_idx]}: {count} files")
    
    # Load model
    print(f"\nLoading model from {args.model_checkpoint}")
    model = load_emotion_model(args.model_checkpoint, device)
    
    # Create data loader
    print("Creating data loader...")
    dataset = EmoSoundscapesDataset()
    sampler = EmotionValidateSampler(args.dataset_path, args.batch_size, train_ratio=0.0)  # Use all data
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=emotion_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Extract activations
    print("Extracting activations from model...")
    activations, extracted_audio_names = extract_activations_from_model(model, data_loader, device)
    
    # Compute steering signals
    print("Computing steering signals...")
    steering_signals = compute_category_steering_signals(activations, categories, category_info)
    
    # Save steering signals
    print(f"Saving steering signals to {args.output_dir}")
    save_steering_signals(steering_signals, category_info, args.output_dir)
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_categories(valence, arousal, categories, category_info, args.output_dir)
    
    print(f"\nSteering signals generation completed!")
    print(f"Output directory: {args.output_dir}")
    print(f"Generated {len(steering_signals)} steering signal categories")


def load_emotion_model(checkpoint_path, device):
    """
    Load emotion model from checkpoint.
    
    Args:
        checkpoint_path: str, path to model checkpoint
        device: torch.device, device to load model on
        
    Returns:
        model: loaded emotion model
    """
    # Load checkpoint with weights_only=False to handle older checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine model type from checkpoint or path
    model_type = None
    if 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
    else:
        # Try to infer from path
        if 'LRM' in checkpoint_path:
            model_type = 'FeatureEmotionRegression_Cnn6_LRM'
        elif 'NewAffective' in checkpoint_path:
            model_type = 'FeatureEmotionRegression_Cnn6_NewAffective'
        else:
            model_type = 'FeatureEmotionRegression_Cnn6_LRM'  # Default
    
    # Create model based on type
    if model_type == 'FeatureEmotionRegression_Cnn6_LRM':
        # Use imported cnn6_config
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
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load model weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    # Move to device
    model = model.to(device)
    
    print(f"Loaded {model_type} model from {checkpoint_path}")
    return model


if __name__ == '__main__':
    main()
