#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

import torch
import numpy as np
import h5py
from scipy.stats import pearsonr
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import warnings
import random
warnings.filterwarnings('ignore')

# Set clean style
plt.style.use('default')
sns.set_palette("husl")

# Import our modules
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def set_all_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_emotion_features(hdf5_path, split='val', train_ratio=0.7, random_seed=42):
    """Load emotion features from HDF5 file with fixed seed."""
    # Set seed for reproducible split
    np.random.seed(random_seed)
    
    with h5py.File(hdf5_path, 'r') as hf:
        features = hf['feature'][:]
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        audio_names = [name.decode() if isinstance(name, bytes) else name for name in hf['audio_name'][:]]
    
    # Create train/val split with fixed seed
    total_samples = len(features)
    train_size = int(total_samples * train_ratio)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    if split == 'val':
        selected_indices = indices[train_size:]
    else:
        selected_indices = indices
    
    selected_features = features[selected_indices]
    selected_labels = np.column_stack([valence[selected_indices], arousal[selected_indices]])
    
    return selected_features, selected_labels, selected_indices

def load_steering_signals_json(json_path):
    """Load steering signals from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    steering_signals = {}
    for category, signals in data.items():
        if category in ['metadata', 'generation_config']:
            continue
            
        steering_signals[category] = {
            'valence': torch.tensor(signals['valence_128d'], dtype=torch.float32),
            'arousal': torch.tensor(signals['arousal_128d'], dtype=torch.float32)
        }
    return steering_signals

def categorize_emotion_25bin(valence, arousal):
    """Categorize emotion into 25-bin system (5x5 grid)."""
    # Valence categories
    if valence <= -0.6:
        v_cat = 'very_negative'
    elif valence <= -0.2:
        v_cat = 'negative'
    elif valence <= 0.2:
        v_cat = 'neutral'
    elif valence <= 0.6:
        v_cat = 'positive'
    else:
        v_cat = 'very_positive'
    
    # Arousal categories  
    if arousal <= -0.6:
        a_cat = 'very_weak'
    elif arousal <= -0.2:
        a_cat = 'weak'
    elif arousal <= 0.2:
        a_cat = 'middle'
    elif arousal <= 0.6:
        a_cat = 'strong'
    else:
        a_cat = 'very_strong'
    
    return f"{v_cat}_{a_cat}"

def evaluate_steering_strength_reproducible(model, features, labels, steering_signals, strength, num_samples=None, seed=42):
    """Evaluate steering performance with fixed sample selection for reproducibility."""
    predictions = []
    device = next(model.parameters()).device
    
    # Move steering signals to device
    for category in steering_signals:
        steering_signals[category]['valence'] = steering_signals[category]['valence'].to(device)
        steering_signals[category]['arousal'] = steering_signals[category]['arousal'].to(device)
    
    # Use ALL samples if num_samples is None, otherwise use subset
    if num_samples is None:
        sample_indices = np.arange(len(features))  # Use all samples
        print(f"    Using ALL {len(features)} validation samples")
    else:
        # Set seed for reproducible sample selection
        np.random.seed(seed + int(strength * 10))  # Different seed per strength but reproducible
        sample_indices = np.random.choice(len(features), min(num_samples, len(features)), replace=False)
        print(f"    Using {len(sample_indices)} random samples")
    
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc=f"Strength {strength:.1f}", leave=False):
            try:
                sample = torch.tensor(features[idx:idx+1], dtype=torch.float32).to(device)
                target_valence, target_arousal = labels[idx]
                
                # Get steering signal based on target emotion
                category = categorize_emotion_25bin(target_valence, target_arousal)
                
                if category in steering_signals:
                    valence_signal = steering_signals[category]['valence']
                    arousal_signal = steering_signals[category]['arousal']
                else:
                    # Fallback to neutral_middle
                    fallback_category = 'neutral_middle' if 'neutral_middle' in steering_signals else list(steering_signals.keys())[0]
                    valence_signal = steering_signals[fallback_category]['valence']
                    arousal_signal = steering_signals[fallback_category]['arousal']
                
                # Create steering signals with specified strength
                steering_signals_list = [
                    {'source': 'affective_valence_128d', 'activation': valence_signal, 'strength': strength, 'alpha': 1.0},
                    {'source': 'affective_arousal_128d', 'activation': arousal_signal, 'strength': strength, 'alpha': 1.0}
                ]
                
                # Forward pass with steering
                model.lrm.enable()
                output = model(sample, forward_passes=2, steering_signals=steering_signals_list, first_pass_steering=True)
                
                pred_valence = output['valence'].cpu().numpy()[0, 0]
                pred_arousal = output['arousal'].cpu().numpy()[0, 0]
                
                # Check for valid predictions
                if not (np.isnan(pred_valence) or np.isnan(pred_arousal) or 
                       abs(pred_valence) > 10.0 or abs(pred_arousal) > 10.0):
                    predictions.append([pred_valence, pred_arousal])
                    
            except Exception:
                continue
    
    if len(predictions) == 0:
        return 0.0, 0.0
    
    predictions = np.array(predictions)
    selected_labels = labels[sample_indices[:len(predictions)]]
    
    # Calculate correlations
    try:
        valence_corr = float(pearsonr(selected_labels[:, 0], predictions[:, 0])[0])
        arousal_corr = float(pearsonr(selected_labels[:, 1], predictions[:, 1])[0])
        
        if np.isnan(valence_corr):
            valence_corr = 0.0
        if np.isnan(arousal_corr):
            arousal_corr = 0.0
            
    except Exception:
        valence_corr = 0.0
        arousal_corr = 0.0
    
    return valence_corr, arousal_corr

def evaluate_baseline_reproducible(model, features, labels, num_samples=None, seed=42):
    """Evaluate baseline performance with fixed sample selection."""
    predictions = []
    device = next(model.parameters()).device
    
    # Use ALL samples if num_samples is None, otherwise use subset
    if num_samples is None:
        sample_indices = np.arange(len(features))  # Use all samples
        print(f"    Using ALL {len(features)} validation samples for baseline")
    else:
        # Set seed for reproducible sample selection
        np.random.seed(seed)
        sample_indices = np.random.choice(len(features), min(num_samples, len(features)), replace=False)
        print(f"    Using {len(sample_indices)} random samples for baseline")
    
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc="Baseline", leave=False):
            sample = torch.tensor(features[idx:idx+1], dtype=torch.float32).to(device)
            
            model.lrm.enable()
            output = model(sample, forward_passes=2)
            
            pred_valence = output['valence'].cpu().numpy()[0, 0]
            pred_arousal = output['arousal'].cpu().numpy()[0, 0]
            predictions.append([pred_valence, pred_arousal])
    
    predictions = np.array(predictions)
    selected_labels = labels[sample_indices]
    
    # Calculate correlations
    valence_corr = float(pearsonr(selected_labels[:, 0], predictions[:, 0])[0])
    arousal_corr = float(pearsonr(selected_labels[:, 1], predictions[:, 1])[0])
    
    return valence_corr, arousal_corr

def create_simple_performance_plot(strengths, valence_corrs, arousal_corrs, baseline_valence, baseline_arousal, output_path):
    """Create a simple, clean plot showing valence and arousal performance vs steering strength."""
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Colors
    valence_color = '#2E86AB'  # Blue
    arousal_color = '#A23B72'  # Pink/Red
    
    # Plot data points
    plt.scatter(strengths[1:], valence_corrs[1:], color=valence_color, s=80, alpha=0.7, 
               label='Valence', zorder=5, edgecolors='white', linewidth=1)
    plt.scatter(strengths[1:], arousal_corrs[1:], color=arousal_color, s=80, alpha=0.7, 
               label='Arousal', zorder=5, edgecolors='white', linewidth=1, marker='s')
    
    # Plot smooth interpolated curves
    try:
        # Create smooth curves for better visualization
        strengths_log = np.log10(strengths[1:])
        strengths_smooth = np.logspace(np.log10(strengths[1]), np.log10(strengths[-1]), 200)
        strengths_smooth_log = np.log10(strengths_smooth)
        
        # Interpolate
        valence_smooth = make_interp_spline(strengths_log, valence_corrs[1:], k=3)
        arousal_smooth = make_interp_spline(strengths_log, arousal_corrs[1:], k=3)
        
        valence_interp = valence_smooth(strengths_smooth_log)
        arousal_interp = arousal_smooth(strengths_smooth_log)
        
        plt.plot(strengths_smooth, valence_interp, color=valence_color, linewidth=3, alpha=0.8)
        plt.plot(strengths_smooth, arousal_interp, color=arousal_color, linewidth=3, alpha=0.8)
    except:
        # Fallback to simple line plot
        plt.plot(strengths[1:], valence_corrs[1:], color=valence_color, linewidth=3, alpha=0.8)
        plt.plot(strengths[1:], arousal_corrs[1:], color=arousal_color, linewidth=3, alpha=0.8)
    
    # Baseline lines
    plt.axhline(y=baseline_valence, color=valence_color, linestyle='--', linewidth=2, alpha=0.6,
               label=f'Baseline Valence (r={baseline_valence:.3f})')
    plt.axhline(y=baseline_arousal, color=arousal_color, linestyle='--', linewidth=2, alpha=0.6,
               label=f'Baseline Arousal (r={baseline_arousal:.3f})')
    
    # Highlight peak performance
    best_valence_idx = np.argmax(valence_corrs[1:])
    best_arousal_idx = np.argmax(arousal_corrs[1:])
    
    plt.scatter([strengths[best_valence_idx + 1]], [valence_corrs[best_valence_idx + 1]], 
               color='gold', s=200, marker='*', zorder=10, edgecolors='black', linewidth=2,
               label=f'Peak Valence (Strength={strengths[best_valence_idx + 1]:.0f})')
    plt.scatter([strengths[best_arousal_idx + 1]], [arousal_corrs[best_arousal_idx + 1]], 
               color='gold', s=200, marker='*', zorder=10, edgecolors='black', linewidth=2,
               label=f'Peak Arousal (Strength={strengths[best_arousal_idx + 1]:.0f})')
    
    # Formatting
    plt.xscale('log')
    plt.xlabel('Steering Strength (log scale)', fontsize=14, fontweight='bold')
    plt.ylabel('Pearson Correlation', fontsize=14, fontweight='bold')
    plt.title('25-Bin Steering Signal Performance vs Strength', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Set reasonable y-axis limits
    all_corrs = valence_corrs + arousal_corrs
    y_min = min(all_corrs) - 0.02
    y_max = max(all_corrs) + 0.02
    plt.ylim(y_min, y_max)
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    return strengths[best_valence_idx + 1], strengths[best_arousal_idx + 1]

def main():
    print("📊 REPRODUCIBLE Steering Performance Curve")
    print("🔒 Fixed seeds ensure identical results every run")
    print("🎯 Creating clean valence/arousal performance plot")
    print("=" * 60)
    
    # SET ALL SEEDS FOR REPRODUCIBILITY
    MASTER_SEED = 42
    set_all_seeds(MASTER_SEED)
    print(f"🔒 All random seeds set to: {MASTER_SEED}")
    
    # Load model
    print("📥 Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, freeze_base=True, forward_passes=2
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = 'workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Load data with fixed seed
    print("📊 Loading validation data...")
    features_path = 'workspaces/emotion_feedback/features/emotion_features.h5'
    features, labels, indices = load_emotion_features(features_path, split='val', random_seed=MASTER_SEED)
    print(f"✅ Loaded {len(features)} validation samples")
    
    # Load steering signals
    print("🎛️ Loading 25-bin steering signals...")
    steering_signals = load_steering_signals_json('tmp/25bin_steering_signals/steering_signals_25bin.json')
    print(f"✅ Loaded {len(steering_signals)} categories")
    
    # Key strength points for clean curve
    strengths = [
        0.0,      # Baseline
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0,
        12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 75.0, 100.0,
        150.0, 200.0, 300.0, 500.0, 1000.0, 2000.0
    ]
    
    # Use ALL validation samples for most accurate results
    num_samples = None  # None means use all samples
    
    print(f"\n📊 Testing {len(strengths)} strength points")
    print(f"🎯 Using ALL {len(features)} validation samples per strength")
    print(f"🔒 Results will be IDENTICAL every run")
    print(f"⚠️  This will take longer but give most accurate results")
    
    # Evaluate baseline with all samples
    print("\n🎯 Evaluating baseline...")
    baseline_valence, baseline_arousal = evaluate_baseline_reproducible(
        model, features, labels, num_samples, MASTER_SEED)
    
    # Evaluate all strengths using all samples
    print("\n📊 Evaluating steering strengths...")
    valence_corrs = [baseline_valence]  # Start with baseline
    arousal_corrs = [baseline_arousal]
    
    for strength in tqdm(strengths[1:], desc="Evaluating strengths"):
        val_corr, ar_corr = evaluate_steering_strength_reproducible(
            model, features, labels, steering_signals, strength, num_samples, MASTER_SEED)
        valence_corrs.append(val_corr)
        arousal_corrs.append(ar_corr)
    
    # Create simple plot
    print("\n📊 Creating performance plot...")
    output_path = 'tmp/reproducible_steering_performance_curve.png'
    best_val_strength, best_ar_strength = create_simple_performance_plot(
        strengths, valence_corrs, arousal_corrs, baseline_valence, baseline_arousal, output_path)
    
    # Print key results
    print("\n" + "="*70)
    print("📊 REPRODUCIBLE STEERING PERFORMANCE SUMMARY")
    print("="*70)
    print(f"🔒 Master Seed: {MASTER_SEED} (results will be identical every run)")
    print(f"📊 Baseline Performance:")
    print(f"   Valence: r = {baseline_valence:.3f}")
    print(f"   Arousal: r = {baseline_arousal:.3f}")
    print(f"\n🏆 Peak Performance:")
    print(f"   Best Valence: Strength {best_val_strength:.0f}, r = {max(valence_corrs[1:]):.3f} (+{(max(valence_corrs[1:]) - baseline_valence)/baseline_valence*100:.1f}%)")
    print(f"   Best Arousal: Strength {best_ar_strength:.0f}, r = {max(arousal_corrs[1:]):.3f} (+{(max(arousal_corrs[1:]) - baseline_arousal)/baseline_arousal*100:.1f}%)")
    
    # Save results with seed info
    results = {
        'master_seed': MASTER_SEED,
        'reproducible': True,
        'strengths': [float(s) for s in strengths],
        'valence_correlations': [float(v) for v in valence_corrs],
        'arousal_correlations': [float(a) for a in arousal_corrs],
        'baseline_valence': float(baseline_valence),
        'baseline_arousal': float(baseline_arousal),
        'best_valence_strength': float(best_val_strength),
        'best_arousal_strength': float(best_ar_strength)
    }
    
    with open('tmp/reproducible_steering_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Clean performance plot saved to: {output_path}")
    print(f"📊 Reproducible results saved to: tmp/reproducible_steering_results.json")
    print(f"🔒 Run this script again - you'll get IDENTICAL results!")
    print("\n✅ Reproducible analysis complete!")

if __name__ == "__main__":
    main() 