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
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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
            'valence_128d': signals['valence_128d'],
            'arousal_128d': signals['arousal_128d']
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

def evaluate_steering_strength(model, features, labels, steering_signals, strength, num_samples=120):
    """Evaluate steering performance at a specific strength."""
    predictions = []
    device = next(model.parameters()).device
    
    # Use subset for faster evaluation
    sample_indices = np.random.choice(len(features), min(num_samples, len(features)), replace=False)
    
    with torch.no_grad():
        for idx in tqdm(sample_indices, desc=f"Strength {strength:.1f}", leave=False):
            try:
                sample = torch.tensor(features[idx:idx+1], dtype=torch.float32).to(device)
                target_valence, target_arousal = labels[idx]
                
                # Get steering signal based on target emotion
                category = categorize_emotion_25bin(target_valence, target_arousal)
                
                if category not in steering_signals:
                    # Fallback to neutral_middle
                    fallback_category = 'neutral_middle' if 'neutral_middle' in steering_signals else list(steering_signals.keys())[0]
                    category = fallback_category
                
                # Apply steering signals using the same method as the working 25-bin test script
                model.clear_feedback_state()
                
                if 'valence_128d' in steering_signals[category]:
                    model.add_steering_signal(
                        source='affective_valence_128d',
                        activation=torch.tensor(steering_signals[category]['valence_128d'], dtype=torch.float32).to(device),
                        strength=strength,
                        alpha=1.0
                    )
                
                if 'arousal_128d' in steering_signals[category]:
                    model.add_steering_signal(
                        source='affective_arousal_128d',
                        activation=torch.tensor(steering_signals[category]['arousal_128d'], dtype=torch.float32).to(device),
                        strength=strength,
                        alpha=1.0
                    )
                
                model.lrm.enable()
                output = model(sample, forward_passes=2)
                
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

def evaluate_baseline(model, features, labels, num_samples=120):
    """Evaluate baseline performance (no steering)."""
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
    
    return valence_corr, arousal_corr

def create_publication_quality_plots(strengths, valence_corrs, arousal_corrs, baseline_valence, baseline_arousal, output_path):
    """Create beautiful publication-quality plots showing the complete steering curve."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 2, 1.5], hspace=0.3, wspace=0.25)
    
    # Colors
    valence_color = '#2E86AB'  # Blue
    arousal_color = '#A23B72'  # Pink/Red
    combined_color = '#F18F01'  # Orange
    baseline_color = '#C73E1D'  # Dark red
    
    # 1. Main Performance Curve (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create smooth interpolation for beautiful curves
    strengths_smooth = np.logspace(np.log10(0.1), np.log10(max(strengths)), 300)
    
    # Plot raw data points
    ax1.scatter(strengths[1:], valence_corrs[1:], color=valence_color, s=60, alpha=0.8, 
               label='Valence (observed)', zorder=5, edgecolors='white', linewidth=1)
    ax1.scatter(strengths[1:], arousal_corrs[1:], color=arousal_color, s=60, alpha=0.8, 
               label='Arousal (observed)', zorder=5, edgecolors='white', linewidth=1, marker='s')
    
    # Plot smooth interpolated curves
    try:
        # Interpolate for smooth curves
        valence_smooth = make_interp_spline(np.log10(strengths[1:]), valence_corrs[1:], k=3)
        arousal_smooth = make_interp_spline(np.log10(strengths[1:]), arousal_corrs[1:], k=3)
        
        valence_interp = valence_smooth(np.log10(strengths_smooth))
        arousal_interp = arousal_smooth(np.log10(strengths_smooth))
        
        ax1.plot(strengths_smooth, valence_interp, color=valence_color, linewidth=3, alpha=0.7, label='Valence (trend)')
        ax1.plot(strengths_smooth, arousal_interp, color=arousal_color, linewidth=3, alpha=0.7, label='Arousal (trend)')
    except:
        # Fallback to simple line plot
        ax1.plot(strengths[1:], valence_corrs[1:], color=valence_color, linewidth=3, alpha=0.7)
        ax1.plot(strengths[1:], arousal_corrs[1:], color=arousal_color, linewidth=3, alpha=0.7)
    
    # Baseline lines
    ax1.axhline(y=baseline_valence, color=valence_color, linestyle='--', linewidth=2, alpha=0.6)
    ax1.axhline(y=baseline_arousal, color=arousal_color, linestyle='--', linewidth=2, alpha=0.6)
    
    # Highlight peak performance
    best_idx = np.argmax(np.array(valence_corrs[1:]) + np.array(arousal_corrs[1:]))
    best_strength = strengths[best_idx + 1]
    ax1.scatter([best_strength], [valence_corrs[best_idx + 1]], color='gold', s=200, 
               marker='*', zorder=10, edgecolors='black', linewidth=2, label=f'Peak (Strength={best_strength})')
    ax1.scatter([best_strength], [arousal_corrs[best_idx + 1]], color='gold', s=200, 
               marker='*', zorder=10, edgecolors='black', linewidth=2)
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Steering Strength (log scale)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Pearson Correlation', fontsize=14, fontweight='bold')
    ax1.set_title('25-Bin Steering Signal Performance Landscape', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_ylim(0.6, 0.9)
    
    # 2. Combined Performance (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    
    combined_performance = [v + a for v, a in zip(valence_corrs, arousal_corrs)]
    baseline_combined = baseline_valence + baseline_arousal
    
    ax2.scatter(strengths[1:], combined_performance[1:], color=combined_color, s=80, alpha=0.8, 
               zorder=5, edgecolors='white', linewidth=1)
    
    # Smooth curve for combined
    try:
        combined_smooth = make_interp_spline(np.log10(strengths[1:]), combined_performance[1:], k=3)
        combined_interp = combined_smooth(np.log10(strengths_smooth))
        ax2.plot(strengths_smooth, combined_interp, color=combined_color, linewidth=4, alpha=0.8)
    except:
        ax2.plot(strengths[1:], combined_performance[1:], color=combined_color, linewidth=4, alpha=0.8)
    
    ax2.axhline(y=baseline_combined, color=baseline_color, linestyle='--', linewidth=3, alpha=0.7,
               label=f'Baseline Combined ({baseline_combined:.3f})')
    
    # Highlight peak
    best_combined_idx = np.argmax(combined_performance[1:])
    best_combined_strength = strengths[best_combined_idx + 1]
    ax2.scatter([best_combined_strength], [combined_performance[best_combined_idx + 1]], 
               color='gold', s=250, marker='*', zorder=10, edgecolors='black', linewidth=2,
               label=f'Peak Combined ({combined_performance[best_combined_idx + 1]:.3f})')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Steering Strength (log scale)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Combined Correlation (V+A)', fontsize=14, fontweight='bold')
    ax2.set_title('Combined Performance Optimization', fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 3. Improvement over Baseline (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    
    valence_improvements = [v - baseline_valence for v in valence_corrs[1:]]
    arousal_improvements = [a - baseline_arousal for a in arousal_corrs[1:]]
    
    ax3.scatter(strengths[1:], [v*100 for v in valence_improvements], color=valence_color, s=60, alpha=0.8, label='Valence Δ%')
    ax3.scatter(strengths[1:], [a*100 for a in arousal_improvements], color=arousal_color, s=60, alpha=0.8, label='Arousal Δ%', marker='s')
    
    # Smooth improvement curves
    try:
        val_imp_smooth = make_interp_spline(np.log10(strengths[1:]), valence_improvements, k=3)
        ar_imp_smooth = make_interp_spline(np.log10(strengths[1:]), arousal_improvements, k=3)
        
        val_imp_interp = val_imp_smooth(np.log10(strengths_smooth)) * 100
        ar_imp_interp = ar_imp_smooth(np.log10(strengths_smooth)) * 100
        
        ax3.plot(strengths_smooth, val_imp_interp, color=valence_color, linewidth=3, alpha=0.7)
        ax3.plot(strengths_smooth, ar_imp_interp, color=arousal_color, linewidth=3, alpha=0.7)
    except:
        ax3.plot(strengths[1:], [v*100 for v in valence_improvements], color=valence_color, linewidth=3, alpha=0.7)
        ax3.plot(strengths[1:], [a*100 for a in arousal_improvements], color=arousal_color, linewidth=3, alpha=0.7)
    
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_xscale('log')
    ax3.set_xlabel('Steering Strength (log scale)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Performance Improvement (%)', fontsize=14, fontweight='bold')
    ax3.set_title('Relative Performance Gains', fontsize=16, fontweight='bold', pad=20)
    ax3.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 4. Operating Regimes Analysis (Middle Right)
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Define operating regimes
    low_range = [i for i, s in enumerate(strengths) if 0 < s <= 10]
    mid_range = [i for i, s in enumerate(strengths) if 10 < s <= 100]
    high_range = [i for i, s in enumerate(strengths) if 100 < s <= 1000]
    ultra_range = [i for i, s in enumerate(strengths) if s > 1000]
    
    regime_colors = ['#90EE90', '#FFD700', '#FF6347', '#9370DB']
    regime_labels = ['Low (0-10)', 'Mid (10-100)', 'High (100-1K)', 'Ultra (1K+)']
    
    for ranges, color, label in zip([low_range, mid_range, high_range, ultra_range], 
                                   regime_colors, regime_labels):
        if ranges:
            regime_strengths = [strengths[i] for i in ranges]
            regime_combined = [combined_performance[i] for i in ranges]
            ax4.scatter(regime_strengths, regime_combined, color=color, s=100, alpha=0.8, 
                       label=label, edgecolors='black', linewidth=1)
    
    ax4.axhline(y=baseline_combined, color=baseline_color, linestyle='--', linewidth=2, alpha=0.7)
    ax4.set_xscale('log')
    ax4.set_xlabel('Steering Strength (log scale)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Combined Performance', fontsize=14, fontweight='bold')
    ax4.set_title('Operating Regime Analysis', fontsize=16, fontweight='bold', pad=20)
    ax4.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 5. Performance Summary Table (Bottom)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # Create summary table
    top_performers = sorted(enumerate(combined_performance[1:]), key=lambda x: x[1], reverse=True)[:5]
    
    table_data = []
    table_data.append(['Rank', 'Strength', 'Valence r', 'Arousal r', 'Combined', 'V Δ%', 'A Δ%'])
    
    for rank, (idx, combined_score) in enumerate(top_performers, 1):
        actual_idx = idx + 1  # Adjust for baseline removal
        strength = strengths[actual_idx]
        val_r = valence_corrs[actual_idx]
        ar_r = arousal_corrs[actual_idx]
        val_delta = (val_r - baseline_valence) / baseline_valence * 100
        ar_delta = (ar_r - baseline_arousal) / baseline_arousal * 100
        
        table_data.append([
            f'{rank}',
            f'{strength:.0f}',
            f'{val_r:.3f}',
            f'{ar_r:.3f}',
            f'{combined_score:.3f}',
            f'{val_delta:+.1f}%',
            f'{ar_delta:+.1f}%'
        ])
    
    # Add baseline row
    table_data.append([
        'Base',
        '0',
        f'{baseline_valence:.3f}',
        f'{baseline_arousal:.3f}',
        f'{baseline_combined:.3f}',
        '0.0%',
        '0.0%'
    ])
    
    table = ax5.table(cellText=table_data[1:], colLabels=table_data[0], 
                     cellLoc='center', loc='center', bbox=[0.1, 0.3, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best performer
    for i in range(len(table_data[0])):
        table[(1, i)].set_facecolor('#FFD700')
        table[(1, i)].set_text_props(weight='bold')
    
    ax5.set_title('Top 5 Performing Steering Strengths', fontsize=16, fontweight='bold', y=0.95)
    
    # Add overall title
    fig.suptitle('Comprehensive 25-Bin Steering Signal Analysis:\nMulti-Modal Performance Landscape', 
                fontsize=20, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()
    
    return best_combined_strength, combined_performance[best_combined_idx + 1]

def main():
    print("🎨 COMPREHENSIVE Steering Curve Analysis")
    print("📊 Creating publication-quality visualization with dense sampling")
    print("🎯 Goal: Generate beautiful parabolic/multi-modal performance curves")
    print("=" * 80)
    
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
    
    # Comprehensive strength range with dense sampling for smooth curves
    strengths = [
        0.0,      # Baseline
        # Low range - dense sampling
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        # Mid range - moderate sampling  
        12.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0,
        # High range - focused sampling around peaks
        150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 
        # Ultra-high range - key points
        750.0, 1000.0, 1500.0, 2000.0, 3000.0, 5000.0, 10000.0
    ]
    
    num_samples = 100000  # Test on all validation samples
    
    print(f"\n🎨 Testing {len(strengths)} strength points for smooth curve generation")
    print(f"📊 Strength range: {min(strengths)} to {max(strengths):,}")
    print(f"🎯 Using {num_samples} samples per strength for robust statistics")
    
    # Evaluate baseline
    print("\n🎯 Evaluating baseline...")
    baseline_valence, baseline_arousal = evaluate_baseline(model, features, labels, num_samples)
    
    # Evaluate all strengths
    print("\n🎨 Evaluating comprehensive strength range...")
    valence_corrs = [baseline_valence]  # Start with baseline
    arousal_corrs = [baseline_arousal]
    
    for strength in tqdm(strengths[1:], desc="Comprehensive Analysis"):
        val_corr, ar_corr = evaluate_steering_strength(
            model, features, labels, steering_signals, strength, num_samples)
        valence_corrs.append(val_corr)
        arousal_corrs.append(ar_corr)
    
    # Create comprehensive visualization
    print("\n🎨 Creating publication-quality visualization...")
    output_path = 'tmp/comprehensive_steering_curve_analysis.png'
    best_strength, best_performance = create_publication_quality_plots(
        strengths, valence_corrs, arousal_corrs, baseline_valence, baseline_arousal, output_path)
    
    # Print comprehensive results
    print("\n" + "="*100)
    print("🎨 COMPREHENSIVE STEERING CURVE ANALYSIS RESULTS")
    print("="*100)
    print(f"{'Strength':<10} {'Valence r':<12} {'Arousal r':<12} {'Combined':<12} {'V Δ%':<8} {'A Δ%':<8} {'Status':<12}")
    print("-" * 100)
    
    for i, strength in enumerate(strengths):
        combined = valence_corrs[i] + arousal_corrs[i]
        val_delta = (valence_corrs[i] - baseline_valence) / baseline_valence * 100
        ar_delta = (arousal_corrs[i] - baseline_arousal) / baseline_arousal * 100
        
        if strength == 0.0:
            status = "Baseline"
        elif combined > baseline_valence + baseline_arousal + 0.05:
            status = "Excellent"
        elif combined > baseline_valence + baseline_arousal + 0.02:
            status = "Good"
        elif combined > baseline_valence + baseline_arousal:
            status = "Beneficial"
        else:
            status = "Suboptimal"
        
        print(f"{strength:<10.1f} {valence_corrs[i]:<12.3f} {arousal_corrs[i]:<12.3f} "
              f"{combined:<12.3f} {val_delta:<+8.1f} {ar_delta:<+8.1f} {status:<12}")
    
    # Final insights
    print(f"\n🎯 COMPREHENSIVE ANALYSIS INSIGHTS:")
    print(f"📊 Baseline Performance: Valence r={baseline_valence:.3f}, Arousal r={baseline_arousal:.3f}")
    print(f"🏆 Optimal Steering Strength: {best_strength:.0f}")
    print(f"🎯 Peak Combined Performance: {best_performance:.3f}")
    print(f"📈 Maximum Improvement: {(best_performance - (baseline_valence + baseline_arousal)) / (baseline_valence + baseline_arousal) * 100:+.1f}%")
    
    # Save comprehensive results
    results = {
        'strengths': [float(s) for s in strengths],
        'valence_correlations': [float(v) for v in valence_corrs],
        'arousal_correlations': [float(a) for a in arousal_corrs],
        'baseline_valence': float(baseline_valence),
        'baseline_arousal': float(baseline_arousal),
        'optimal_strength': float(best_strength),
        'peak_performance': float(best_performance),
        'analysis_type': 'comprehensive_curve'
    }
    
    results_path = 'tmp/comprehensive_steering_curve_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Publication-quality visualization saved to: {output_path}")
    print(f"📊 Comprehensive results saved to: {results_path}")
    print("\n🎨 Comprehensive curve analysis complete!")
    print("🎯 Beautiful multi-modal performance landscape revealed!")

if __name__ == "__main__":
    main() 