#!/usr/bin/env python3

import sys
import os
# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import json
import numpy as np
import h5py
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

def load_real_emotion_data(hdf5_path, split='val', train_ratio=0.7, max_samples=None):
    """Load real emotion features from HDF5 file."""
    print(f"🔍 Loading real emotion data from: {hdf5_path}")
    
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
    real_data = []
    for i in range(len(selected_features)):
        real_data.append((
            torch.tensor(selected_features[i], dtype=torch.float32),
            selected_valence[i],
            selected_arousal[i],
            selected_names[i]
        ))
    
    return real_data

def test_arousal_only_steering():
    """Test steering ONLY the arousal target layers (conv3/conv4)."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, forward_passes=2
    ).to(device)
    
    # Load newly trained model with corrected steering
    checkpoint_path = '/home/pengliu/Private/my_panns_transfer_to_gtzan/workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()  # CRITICAL: Set to eval mode
        print("✅ Loaded newly trained model with corrected steering")
    else:
        print(f"❌ Error: New model not found at {checkpoint_path}")
        return
    
    # Load 9-bin steering signals
    steering_signals_path = './steering_signal_pairs_9bin/steering_signal_pairs_9bin.json'
    if not os.path.exists(steering_signals_path):
        print(f"❌ Error: 9-bin steering signals not found at {steering_signals_path}")
        return
    
    with open(steering_signals_path, 'r') as f:
        signals_9bin = json.load(f)
    
    # Check if we loaded the 9-bin format and count categories properly
    excluded_keys = {'metadata', 'generation_config'}
    category_count = len([k for k in signals_9bin.keys() if k not in excluded_keys])
    
    if 'metadata' in signals_9bin and 'binning_system' in signals_9bin.get('metadata', {}):
        binning = signals_9bin['metadata']['binning_system']
        method = signals_9bin['metadata']['extraction_method']
        print(f"✅ Loaded steering signals: {category_count} categories (Method: {method}, Binning: {binning})")
        print("   🎯 Using 9-bin steering signals with AROUSAL-ONLY targeting!")
    else:
        print(f"✅ Loaded steering signals: {category_count} categories (Legacy format)")
    
    # Load REAL emotion dataset
    dataset_paths = [
        'workspaces/emotion_feedback/features/emotion_features.h5',
    ]
    
    real_data = None
    for path in dataset_paths:
        if os.path.exists(path):
            real_data = load_real_emotion_data(path, split='val', max_samples=None)  # Use ALL validation samples
            break
    
    if real_data is None:
        print("❌ Error: Could not find real emotion dataset in any of these locations:")
        for path in dataset_paths:
            print(f"   - {path}")
        return
    
    print(f"✅ Loaded {len(real_data)} REAL emotion validation samples")
    
    # Analyze what categories we have (9-bin)
    print("\n🔍 Analyzing target categories in dataset (9-bin)...")
    category_counts = {}
    for features, true_val, true_aro, audio_name in real_data:
        target_category = categorize_emotion_9bin(true_val, true_aro)
        category_counts[target_category] = category_counts.get(target_category, 0) + 1
    
    print(f"Found {len(category_counts)} unique categories (9-bin):")
    for category, count in sorted(category_counts.items()):
        available = "✅" if category in signals_9bin and category not in excluded_keys else "❌"
        print(f"   {available} {category}: {count} samples")
    
    # Test different strengths
    strengths = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    results = {}
    
    for strength in strengths:
        print(f"\n🔍 Testing AROUSAL-ONLY steering with strength: {strength}")
        
        predictions = []
        steering_count = 0
        missing_count = 0
        
        for sample_idx, (features, true_val, true_aro, audio_name) in enumerate(real_data):
            # Prepare input
            features = features.unsqueeze(0).to(device)  # [1, time_steps, mel_bins]
            
            # Get TARGET CATEGORY for this specific sample (9-bin)
            target_category = categorize_emotion_9bin(true_val, true_aro)
            
            # Prepare steering signals (AROUSAL-ONLY, targeting conv3/conv4)
            if strength > 0.0 and target_category in signals_9bin and target_category not in excluded_keys:
                signals = signals_9bin[target_category]
                steering_signals_current = []
                
                # ONLY use arousal signal, target conv3/conv4 layers
                if 'arousal_128d' in signals:
                    arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
                    
                    # Target conv3 layer (arousal pathway)
                    steering_signals_current.append({
                        'source': 'affective_arousal_128d',
                        'activation': arousal_signal,
                        'strength': strength,
                        'alpha': 1.0
                    })
                    
                    # Target conv4 layer (arousal pathway) - if it exists
                    # Note: We're using the same signal for both conv3 and conv4
                    # This might need adjustment based on actual layer names
                
                steering_count += 1
                
                if sample_idx < 3:  # Debug first few samples
                    print(f"      Sample {sample_idx}: Using TARGET category '{target_category}' (True: V={true_val:.2f}, A={true_aro:.2f})")
                    print(f"         Steering: AROUSAL-ONLY, targeting conv3/conv4 layers")
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
        
        # Calculate correlation (if steering helps, should improve correlation)
        val_corr = np.corrcoef(pred_vals, true_vals)[0, 1] if len(set(pred_vals)) > 1 else 0.0
        aro_corr = np.corrcoef(pred_aros, true_aros)[0, 1] if len(set(pred_aros)) > 1 else 0.0
        
        coverage = steering_count / len(real_data) * 100
        
        results[strength] = {
            'mean_pred_val': mean_pred_val,
            'mean_pred_aro': mean_pred_aro,
            'val_corr': val_corr,
            'aro_corr': aro_corr,
            'coverage': coverage,
            'missing': missing_count
        }
        
        print(f"   Samples: {len(predictions)}, Steering coverage: {coverage:.1f}%, Missing: {missing_count}")
        print(f"   Valence: r={val_corr:.3f}, mean_pred={mean_pred_val:.3f}, mean_true={mean_true_val:.3f}")
        print(f"   Arousal: r={aro_corr:.3f}, mean_pred={mean_pred_aro:.3f}, mean_true={mean_true_aro:.3f}")
    
    # Compare results
    print(f"\n📊 AROUSAL-ONLY Results Comparison:")
    print(f"{'Strength':>8} {'Val r':>8} {'Aro r':>8} {'Val Δr':>8} {'Aro Δr':>8} {'Coverage':>8}")
    print("-" * 60)
    
    baseline = results[0.0]
    for strength in strengths:
        res = results[strength]
        val_delta_r = res['val_corr'] - baseline['val_corr']
        aro_delta_r = res['aro_corr'] - baseline['aro_corr']
        
        print(f"{strength:>8.1f} {res['val_corr']:>8.3f} {res['aro_corr']:>8.3f} "
              f"{val_delta_r:>+8.3f} {aro_delta_r:>+8.3f} {res['coverage']:>7.1f}%")
    
    # Summary
    best_val_improvement = max(results[s]['val_corr'] - baseline['val_corr'] for s in strengths[1:])
    best_aro_improvement = max(results[s]['aro_corr'] - baseline['aro_corr'] for s in strengths[1:])
    
    print(f"\n✅ AROUSAL-ONLY Summary:")
    print(f"   Best valence correlation improvement: {best_val_improvement:+.4f}")
    print(f"   Best arousal correlation improvement: {best_aro_improvement:+.4f}")
    
    if best_aro_improvement > 0.01:
        print("   🎉 SUCCESS: Arousal-only steering improves arousal predictions!")
    elif best_aro_improvement > -0.01:
        print("   ✅ GOOD: Arousal-only steering maintains arousal performance!")
    else:
        print("   ⚠️  MARGINAL: Arousal-only steering shows small effects")
    
    print(f"   Note: Valence performance should remain unchanged (no valence steering)")

if __name__ == "__main__":
    test_arousal_only_steering() 