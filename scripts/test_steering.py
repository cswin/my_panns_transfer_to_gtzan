#!/usr/bin/env python3
"""
Formal Steering Test Script
===========================

This script implements the optimal steering approach based on scientific findings:
- Approach: valence-conv4-only steering
- Performance: +0.014 arousal improvement at strength 2.0
- Coverage: 99.5% of validation samples
- Binning: 9-bin coarse categorization

Based on findings from STEERING_SIGNALS_FINAL_REPORT.md
"""

import sys
import os
import argparse
import json
import numpy as np
import torch
import h5py
from pathlib import Path

# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def categorize_emotion_9bin(valence, arousal):
    """Categorize emotion into 9-bin system (3x3 grid) for optimal sample distribution."""
    
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

def load_emotion_data(hdf5_path, split='val', train_ratio=0.7, max_samples=None):
    """Load emotion features from HDF5 file with train/val split."""
    print(f"🔍 Loading emotion data from: {hdf5_path}")
    
    if not os.path.exists(hdf5_path):
        print(f"❌ Error: Dataset file not found: {hdf5_path}")
        return None
    
    with h5py.File(hdf5_path, 'r') as hf:
        features = hf['feature'][:]
        valence = hf['valence'][:]
        arousal = hf['arousal'][:]
        audio_names = [name.decode() if isinstance(name, bytes) else name for name in hf['audio_name'][:]]
        
        print(f"   Total samples in dataset: {len(features)}")
        print(f"   Feature shape: {features.shape}")
        print(f"   Valence range: [{np.min(valence):.3f}, {np.max(valence):.3f}]")
        print(f"   Arousal range: [{np.min(arousal):.3f}, {np.max(arousal):.3f}]")
    
    # Create train/val split (same as used in training)
    np.random.seed(42)  # Fixed seed for reproducible split
    total_samples = len(features)
    train_size = int(total_samples * train_ratio)
    indices = np.arange(total_samples)
    np.random.shuffle(indices)
    
    if split == 'val':
        selected_indices = indices[train_size:]
    else:
        selected_indices = indices[:train_size]
    
    # Limit to max_samples if specified (None = use all samples)
    if max_samples and len(selected_indices) > max_samples:
        selected_indices = selected_indices[:max_samples]
    
    selected_features = features[selected_indices]
    selected_valence = valence[selected_indices]
    selected_arousal = arousal[selected_indices]
    selected_names = [audio_names[i] for i in selected_indices]
    
    if max_samples:
        print(f"   Selected {len(selected_indices)} {split} samples (limited to {max_samples})")
    else:
        print(f"   Selected {len(selected_indices)} {split} samples (using ALL validation data)")
    
    # Convert to list of tuples for easier processing
    data = []
    for i in range(len(selected_features)):
        data.append((
            torch.tensor(selected_features[i], dtype=torch.float32),
            selected_valence[i],
            selected_arousal[i],
            selected_names[i]
        ))
    
    return data

def test_steering(model, data, steering_signals, strengths, device, output_file=None):
    """
    Test steering with different strengths and approaches.
    
    Args:
        model: The emotion model
        data: List of (features, true_val, true_aro, audio_name) tuples
        steering_signals: Dictionary of steering signals by category
        strengths: List of steering strengths to test
        device: torch device
        output_file: Optional file to save results
    
    Returns:
        Dictionary of results for each strength
    """
    results = {}
    excluded_keys = {'metadata', 'generation_config'}
    
    print(f"\n🔬 Testing steering with {len(strengths)} different strengths...")
    print(f"📊 Dataset size: {len(data)} samples")
    print(f"🎯 Approach: valence-conv4-only steering (optimal approach)")
    print(f"📈 Expected: +0.014 arousal improvement at strength 2.0")
    print()
    
    for strength in strengths:
        print(f"🔍 Testing strength: {strength}")
        
        predictions = []
        steering_count = 0
        missing_count = 0
        
        for sample_idx, (features, true_val, true_aro, audio_name) in enumerate(data):
            # Prepare input
            features = features.unsqueeze(0).to(device)  # [1, time_steps, mel_bins]
            
            # Get target category for this specific sample (9-bin)
            target_category = categorize_emotion_9bin(true_val, true_aro)
            
            # Prepare steering signals (valence-conv4-only, targeting only conv4)
            if strength > 0.0 and target_category in steering_signals and target_category not in excluded_keys:
                signals = steering_signals[target_category]
                steering_signals_current = []
                
                # ONLY use valence signal, target ONLY conv4 layer
                if 'valence_128d' in signals:
                    valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
                    
                    # Target ONLY conv4 layer (valence pathway) - optimal targeting
                    steering_signals_current.append({
                        'source': 'affective_valence_128d',
                        'activation': valence_signal,
                        'strength': strength,
                        'alpha': 1.0
                    })
                
                steering_count += 1
                
                if sample_idx < 3:  # Debug first few samples
                    print(f"      Sample {sample_idx}: Category '{target_category}' (True: V={true_val:.2f}, A={true_aro:.2f})")
            else:
                steering_signals_current = None
                if strength > 0.0:
                    missing_count += 1
            
            # Forward pass
            with torch.no_grad():
                output = model(features, 
                             forward_passes=2,
                             steering_signals=steering_signals_current,
                             first_pass_steering=False)
            
            # Store results
            pred_val = output['valence'][0].item()
            pred_aro = output['arousal'][0].item()
            predictions.append((pred_val, pred_aro, true_val, true_aro, audio_name, target_category))
        
        # Calculate metrics
        pred_vals = [p[0] for p in predictions]
        pred_aros = [p[1] for p in predictions]
        true_vals = [p[2] for p in predictions]
        true_aros = [p[3] for p in predictions]
        
        # Stats
        mean_pred_val = np.mean(pred_vals)
        mean_pred_aro = np.mean(pred_aros)
        mean_true_val = np.mean(true_vals)
        mean_true_aro = np.mean(true_aros)
        
        # Calculate correlation
        val_corr = np.corrcoef(pred_vals, true_vals)[0, 1] if len(set(pred_vals)) > 1 else 0.0
        aro_corr = np.corrcoef(pred_aros, true_aros)[0, 1] if len(set(pred_aros)) > 1 else 0.0
        
        coverage = steering_count / len(data) * 100
        
        results[strength] = {
            'mean_pred_val': mean_pred_val,
            'mean_pred_aro': mean_pred_aro,
            'val_corr': val_corr,
            'aro_corr': aro_corr,
            'coverage': coverage,
            'missing': missing_count,
            'total_samples': len(predictions)
        }
        
        print(f"   Samples: {len(predictions)}, Coverage: {coverage:.1f}%, Missing: {missing_count}")
        print(f"   Valence: r={val_corr:.3f}, mean_pred={mean_pred_val:.3f}, mean_true={mean_true_val:.3f}")
        print(f"   Arousal: r={aro_corr:.3f}, mean_pred={mean_pred_aro:.3f}, mean_true={mean_true_aro:.3f}")
        print()
    
    return results

def print_results_summary(results, strengths):
    """Print a formatted summary of the steering test results."""
    print(f"📊 STEERING TEST RESULTS SUMMARY")
    print(f"=" * 60)
    print(f"{'Strength':>8} {'Val r':>8} {'Aro r':>8} {'Val Δr':>8} {'Aro Δr':>8} {'Coverage':>8}")
    print("-" * 60)
    
    baseline = results[0.0]
    for strength in strengths:
        res = results[strength]
        val_delta_r = res['val_corr'] - baseline['val_corr']
        aro_delta_r = res['aro_corr'] - baseline['aro_corr']
        
        print(f"{strength:>8.1f} {res['val_corr']:>8.3f} {res['aro_corr']:>8.3f} "
              f"{val_delta_r:>+8.3f} {aro_delta_r:>+8.3f} {res['coverage']:>7.1f}%")
    
    # Find best improvements
    best_val_improvement = max(results[s]['val_corr'] - baseline['val_corr'] for s in strengths[1:])
    best_aro_improvement = max(results[s]['aro_corr'] - baseline['aro_corr'] for s in strengths[1:])
    
    best_val_strength = max(strengths[1:], key=lambda s: results[s]['val_corr'] - baseline['val_corr'])
    best_aro_strength = max(strengths[1:], key=lambda s: results[s]['aro_corr'] - baseline['aro_corr'])
    
    print()
    print(f"🏆 BEST PERFORMANCE:")
    print(f"   Valence: +{best_val_improvement:.4f} at strength {best_val_strength}")
    print(f"   Arousal: +{best_aro_improvement:.4f} at strength {best_aro_strength}")
    print()
    
    # Compare with expected results
    print(f"📈 COMPARISON WITH EXPECTED RESULTS:")
    print(f"   Expected arousal improvement: +0.014 at strength 2.0")
    print(f"   Actual arousal improvement: +{best_aro_improvement:.4f} at strength {best_aro_strength}")
    print(f"   Expected coverage: 99.5%")
    print(f"   Actual coverage: {baseline['coverage']:.1f}%")
    
    if best_aro_improvement >= 0.010:
        print(f"   ✅ SUCCESS: Achieved significant arousal improvement!")
    elif best_aro_improvement >= 0.005:
        print(f"   ✅ GOOD: Achieved moderate arousal improvement")
    else:
        print(f"   ⚠️  MARGINAL: Small arousal improvement")
    
    print()

def save_results(results, output_file):
    """Save results to JSON file."""
    if output_file:
        # Convert numpy types to native Python types for JSON serialization
        results_json = {}
        for strength, res in results.items():
            results_json[strength] = {
                'mean_pred_val': float(res['mean_pred_val']),
                'mean_pred_aro': float(res['mean_pred_aro']),
                'val_corr': float(res['val_corr']),
                'aro_corr': float(res['aro_corr']),
                'coverage': float(res['coverage']),
                'missing': int(res['missing']),
                'total_samples': int(res['total_samples'])
            }
        
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"📄 Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Test optimal 9-bin steering approach')
    parser.add_argument('--dataset', type=str, 
                       default='workspaces/emotion_feedback/features/emotion_features.h5',
                       help='Path to emotion dataset HDF5 file')
    parser.add_argument('--model', type=str,
                       default='workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=16/freeze_base=True/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--steering_signals', type=str,
                       default='./steering_signal_pairs_9bin/steering_signal_pairs_9bin.json',
                       help='Path to 9-bin steering signals JSON file')
    parser.add_argument('--output', type=str,
                       default='steering_test_results/steering_results.json',
                       help='Output file for results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to test (None = all)')
    parser.add_argument('--strengths', type=str, default='0.0,1.0,1.5,2.0,2.5,3,3.5,4,4.5,5.0,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10',
                       help='Comma-separated list of steering strengths to test')
    
    args = parser.parse_args()
    
    # Parse strengths
    strengths = [float(s.strip()) for s in args.strengths.split(',')]
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load model
    print(f"📦 Loading model from: {args.model}")
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, forward_passes=2
    ).to(device)
    
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Error: Model checkpoint not found: {args.model}")
        return
    
    # Load steering signals
    print(f"📡 Loading steering signals from: {args.steering_signals}")
    if os.path.exists(args.steering_signals):
        with open(args.steering_signals, 'r') as f:
            steering_signals = json.load(f)
        
        excluded_keys = {'metadata', 'generation_config'}
        category_count = len([k for k in steering_signals.keys() if k not in excluded_keys])
        print(f"✅ Loaded {category_count} steering signal categories")
    else:
        print(f"❌ Error: Steering signals not found: {args.steering_signals}")
        return
    
    # Load dataset
    data = load_emotion_data(args.dataset, split='val', max_samples=args.max_samples)
    if data is None:
        return
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run steering test
    results = test_steering(model, data, steering_signals, strengths, device, args.output)
    
    # Print summary
    print_results_summary(results, strengths)
    
    # Save results
    save_results(results, args.output)
    
    print("🎉 Steering test completed successfully!")

if __name__ == "__main__":
    main() 