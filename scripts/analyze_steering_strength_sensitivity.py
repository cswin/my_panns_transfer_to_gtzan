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

# Import our modules
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def load_emotion_features(hdf5_path, split='val', train_ratio=0.7, random_seed=42):
    """Load emotion features from HDF5 file."""
    with h5py.File(hdf5_path, 'r') as hf:
        features = hf['feature'][:]
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        audio_names = [name.decode() if isinstance(name, bytes) else name for name in hf['audio_name'][:]]
    
    # Create train/val split
    np.random.seed(random_seed)
    total_samples = len(features)
    train_size = int(total_samples * train_ratio)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    if split == 'train':
        selected_indices = indices[:train_size]
    elif split == 'val':
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
        # Skip metadata entries
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

def evaluate_steering_strength(model, features, labels, steering_signals, strength, num_samples=100):
    """Evaluate steering performance at a specific strength."""
    print(f"  🔧 Testing strength: {strength}")
    
    predictions = []
    device = next(model.parameters()).device
    
    # Move steering signals to device
    for category in steering_signals:
        steering_signals[category]['valence'] = steering_signals[category]['valence'].to(device)
        steering_signals[category]['arousal'] = steering_signals[category]['arousal'].to(device)
    
    # Use subset for faster evaluation
    sample_indices = np.random.choice(len(features), min(num_samples, len(features)), replace=False)
    
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc=f"Strength {strength}", leave=False):
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
            predictions.append([pred_valence, pred_arousal])
    
    predictions = np.array(predictions)
    selected_labels = labels[sample_indices]
    
    # Calculate correlations
    valence_corr = float(pearsonr(selected_labels[:, 0], predictions[:, 0])[0])
    arousal_corr = float(pearsonr(selected_labels[:, 1], predictions[:, 1])[0])
    
    return valence_corr, arousal_corr, predictions, selected_labels

def evaluate_baseline(model, features, labels, num_samples=100):
    """Evaluate baseline performance (no steering)."""
    print("  🎯 Testing baseline (no steering)")
    
    predictions = []
    device = next(model.parameters()).device
    
    # Use subset for faster evaluation
    sample_indices = np.random.choice(len(features), min(num_samples, len(features)), replace=False)
    
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
    
    return valence_corr, arousal_corr, predictions, selected_labels

def plot_strength_analysis(strengths, valence_corrs, arousal_corrs, baseline_valence, baseline_arousal, output_path):
    """Create visualization of steering strength analysis."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Correlation vs Strength
    ax1.plot(strengths, valence_corrs, 'b-o', label='Valence', linewidth=2, markersize=6)
    ax1.plot(strengths, arousal_corrs, 'r-s', label='Arousal', linewidth=2, markersize=6)
    ax1.axhline(y=baseline_valence, color='b', linestyle='--', alpha=0.7, label=f'Baseline Valence ({baseline_valence:.3f})')
    ax1.axhline(y=baseline_arousal, color='r', linestyle='--', alpha=0.7, label=f'Baseline Arousal ({baseline_arousal:.3f})')
    ax1.set_xlabel('Steering Strength')
    ax1.set_ylabel('Pearson Correlation')
    ax1.set_title('Steering Performance vs Strength')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Improvement over Baseline
    valence_improvements = np.array(valence_corrs) - baseline_valence
    arousal_improvements = np.array(arousal_corrs) - baseline_arousal
    
    ax2.plot(strengths, valence_improvements, 'b-o', label='Valence Δ', linewidth=2, markersize=6)
    ax2.plot(strengths, arousal_improvements, 'r-s', label='Arousal Δ', linewidth=2, markersize=6)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.set_xlabel('Steering Strength')
    ax2.set_ylabel('Correlation Improvement')
    ax2.set_title('Performance Improvement over Baseline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Relative Improvement (%)
    valence_rel_improvements = (valence_improvements / baseline_valence) * 100
    arousal_rel_improvements = (arousal_improvements / baseline_arousal) * 100
    
    ax3.plot(strengths, valence_rel_improvements, 'b-o', label='Valence %', linewidth=2, markersize=6)
    ax3.plot(strengths, arousal_rel_improvements, 'r-s', label='Arousal %', linewidth=2, markersize=6)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Steering Strength')
    ax3.set_ylabel('Relative Improvement (%)')
    ax3.set_title('Relative Performance Improvement')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Optimal Range Analysis
    # Find optimal strength for each dimension
    best_valence_idx = np.argmax(valence_corrs)
    best_arousal_idx = np.argmax(arousal_corrs)
    best_combined_idx = np.argmax(np.array(valence_corrs) + np.array(arousal_corrs))
    
    ax4.bar(['Valence Optimal', 'Arousal Optimal', 'Combined Optimal'], 
            [strengths[best_valence_idx], strengths[best_arousal_idx], strengths[best_combined_idx]],
            color=['blue', 'red', 'green'], alpha=0.7)
    ax4.set_ylabel('Optimal Strength')
    ax4.set_title('Optimal Steering Strengths')
    ax4.grid(True, alpha=0.3)
    
    # Add text annotations
    ax4.text(0, strengths[best_valence_idx] + 0.1, f'{strengths[best_valence_idx]:.1f}', 
             ha='center', va='bottom', fontweight='bold')
    ax4.text(1, strengths[best_arousal_idx] + 0.1, f'{strengths[best_arousal_idx]:.1f}', 
             ha='center', va='bottom', fontweight='bold')
    ax4.text(2, strengths[best_combined_idx] + 0.1, f'{strengths[best_combined_idx]:.1f}', 
             ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_valence_idx, best_arousal_idx, best_combined_idx

def main():
    print("🎯 25-Bin Steering Strength Sensitivity Analysis")
    print("=" * 60)
    
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
    
    # Load data
    print("📊 Loading validation data...")
    features_path = 'workspaces/emotion_feedback/features/emotion_features.h5'
    features, labels, indices = load_emotion_features(features_path, split='val')
    print(f"✅ Loaded {len(features)} validation samples")
    
    # Load steering signals
    print("🎛️ Loading 25-bin steering signals...")
    steering_signals = load_steering_signals_json('tmp/25bin_steering_signals/steering_signals_25bin.json')
    print(f"✅ Loaded {len(steering_signals)} categories")
    
    # Define strength range to test
    strengths = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0]
    num_samples = 150  # Use subset for faster evaluation
    
    print(f"\n🔬 Testing {len(strengths)} different strengths on {num_samples} samples each")
    print(f"📊 Strength range: {min(strengths)} to {max(strengths)}")
    
    # Evaluate baseline
    print("\n🎯 Evaluating baseline...")
    baseline_valence, baseline_arousal, _, _ = evaluate_baseline(model, features, labels, num_samples)
    
    # Evaluate different strengths
    print("\n🔧 Evaluating steering strengths...")
    valence_corrs = []
    arousal_corrs = []
    
    for strength in tqdm(strengths, desc="Testing strengths"):
        if strength == 0.0:
            # Use baseline results for strength 0
            valence_corrs.append(baseline_valence)
            arousal_corrs.append(baseline_arousal)
        else:
            val_corr, ar_corr, _, _ = evaluate_steering_strength(
                model, features, labels, steering_signals, strength, num_samples)
            valence_corrs.append(val_corr)
            arousal_corrs.append(ar_corr)
    
    # Create analysis
    print("\n📊 Creating strength analysis visualization...")
    output_path = 'tmp/steering_strength_analysis.png'
    best_val_idx, best_ar_idx, best_combined_idx = plot_strength_analysis(
        strengths, valence_corrs, arousal_corrs, baseline_valence, baseline_arousal, output_path)
    
    # Print detailed results
    print("\n" + "="*80)
    print("📊 STEERING STRENGTH SENSITIVITY ANALYSIS RESULTS")
    print("="*80)
    print(f"{'Strength':<10} {'Valence r':<12} {'Arousal r':<12} {'ΔV':<10} {'ΔA':<10} {'Combined':<10}")
    print("-" * 80)
    
    for i, strength in enumerate(strengths):
        delta_v = valence_corrs[i] - baseline_valence
        delta_a = arousal_corrs[i] - baseline_arousal
        combined = valence_corrs[i] + arousal_corrs[i]
        
        print(f"{strength:<10.1f} {valence_corrs[i]:<12.3f} {arousal_corrs[i]:<12.3f} "
              f"{delta_v:<+10.3f} {delta_a:<+10.3f} {combined:<10.3f}")
    
    # Optimal strength analysis
    print(f"\n🎯 OPTIMAL STRENGTH ANALYSIS:")
    print(f"📊 Baseline: Valence r={baseline_valence:.3f}, Arousal r={baseline_arousal:.3f}")
    print(f"🏆 Best Valence: Strength={strengths[best_val_idx]:.1f}, r={valence_corrs[best_val_idx]:.3f} "
          f"(Δ={valence_corrs[best_val_idx]-baseline_valence:+.3f})")
    print(f"🏆 Best Arousal: Strength={strengths[best_ar_idx]:.1f}, r={arousal_corrs[best_ar_idx]:.3f} "
          f"(Δ={arousal_corrs[best_ar_idx]-baseline_arousal:+.3f})")
    print(f"🏆 Best Combined: Strength={strengths[best_combined_idx]:.1f}, "
          f"Total={valence_corrs[best_combined_idx]+arousal_corrs[best_combined_idx]:.3f}")
    
    # Performance insights
    max_val_improvement = float(max(np.array(valence_corrs) - baseline_valence))
    max_ar_improvement = float(max(np.array(arousal_corrs) - baseline_arousal))
    
    print(f"\n📈 PERFORMANCE INSIGHTS:")
    print(f"🎯 Maximum Valence Improvement: +{max_val_improvement:.3f} "
          f"({max_val_improvement/baseline_valence*100:+.1f}%)")
    print(f"🎯 Maximum Arousal Improvement: +{max_ar_improvement:.3f} "
          f"({max_ar_improvement/baseline_arousal*100:+.1f}%)")
    
    # Sensitivity analysis
    effective_strengths = [s for i, s in enumerate(strengths) 
                          if valence_corrs[i] > baseline_valence or arousal_corrs[i] > baseline_arousal]
    
    if effective_strengths:
        print(f"🔧 Effective Strength Range: {min(effective_strengths):.1f} - {max(effective_strengths):.1f}")
        print(f"🎯 Recommended Strength: {strengths[best_combined_idx]:.1f} (best combined performance)")
    else:
        print("⚠️  No strength values showed improvement over baseline")
    
    print(f"\n📊 Visualization saved to: {output_path}")
    
    # Save results to JSON
    results = {
        'strengths': [float(s) for s in strengths],
        'valence_correlations': [float(v) for v in valence_corrs],
        'arousal_correlations': [float(a) for a in arousal_corrs],
        'baseline_valence': float(baseline_valence),
        'baseline_arousal': float(baseline_arousal),
        'optimal_valence_strength': float(strengths[best_val_idx]),
        'optimal_arousal_strength': float(strengths[best_ar_idx]),
        'optimal_combined_strength': float(strengths[best_combined_idx]),
        'max_valence_improvement': max_val_improvement,
        'max_arousal_improvement': max_ar_improvement
    }
    
    results_path = 'tmp/steering_strength_analysis_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Results saved to: {results_path}")

if __name__ == "__main__":
    main() 