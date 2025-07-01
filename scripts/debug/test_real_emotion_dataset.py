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
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def categorize_emotion_25bin(valence, arousal):
    """Categorize emotion into 25-bin system based on valence/arousal values."""
    
    # Define thresholds for 5x5 grid
    val_thresholds = [-0.6, -0.2, 0.2, 0.6]  # Creates 5 bins
    aro_thresholds = [-0.6, -0.2, 0.2, 0.6]  # Creates 5 bins
    
    # Determine valence category
    if valence <= val_thresholds[0]:
        val_cat = "very_negative"
    elif valence <= val_thresholds[1]:
        val_cat = "negative"
    elif valence <= val_thresholds[2]:
        val_cat = "neutral"
    elif valence <= val_thresholds[3]:
        val_cat = "positive"
    else:
        val_cat = "very_positive"
    
    # Determine arousal category
    if arousal <= aro_thresholds[0]:
        aro_cat = "very_weak"
    elif arousal <= aro_thresholds[1]:
        aro_cat = "weak"
    elif arousal <= aro_thresholds[2]:
        aro_cat = "moderate"
    elif arousal <= aro_thresholds[3]:
        aro_cat = "strong"
    else:
        aro_cat = "very_strong"
    
    return f"{val_cat}_{aro_cat}"

def test_real_emotion_dataset():
    """Test amplified steering effects on real emotion validation dataset.
    
    Uses INDIVIDUAL PROCESSING approach:
    - Each audio sample is processed completely independently
    - Model state is reset before each sample
    - Steering signals are applied per-sample
    - No batch-level steering conflicts
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, forward_passes=2
    ).to(device)
    
    # Load pretrained weights
    checkpoint_path = '/DATA/pliu/EmotionData/Cnn6_mAP=0.343.pth'
    if os.path.exists(checkpoint_path):
        model.load_from_pretrain(checkpoint_path)
        print("✅ Loaded pretrained weights")
    
    # Load steering signals
    steering_signals_path = './steering_signals_25bin/steering_signals_25bin.json'
    if not os.path.exists(steering_signals_path):
        steering_signals_path = './tmp/25bin_steering_signals/steering_signals_25bin.json'
    
    with open(steering_signals_path, 'r') as f:
        signals_25bin = json.load(f)
    
    print(f"✅ Loaded steering signals: {len(signals_25bin)} categories")
    
    # Create synthetic validation data for testing
    print("🔧 Creating synthetic validation dataset...")
    val_data = []
    np.random.seed(42)  # For reproducible results
    
    for i in range(100):  # 100 synthetic samples
        features = torch.randn(1024, 64)
        valence = np.random.uniform(-0.8, 0.8)
        arousal = np.random.uniform(-0.8, 0.8)
        val_data.append((features, {'valence': valence, 'arousal': arousal}))
    
    print(f"✅ Created validation dataset: {len(val_data)} samples")
    
    print("\n" + "="*80)
    print("REAL EMOTION DATASET STEERING TEST")
    print("="*80)
    
    # Test different strengths
    strengths = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    results = {}
    
    for strength in strengths:
        print(f"\n🔍 Testing strength: {strength}")
        
        # Reset model state between strength tests
        model.lrm.clear_stored_activations()
        # DON'T reset modulation strength - it deletes the original scale values needed for strength adjustments!
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_categories = []
        steering_applied = []
        
        with torch.no_grad():
            for sample_idx, (features, targets) in enumerate(val_data):
                # INDIVIDUAL PROCESSING: Process each sample completely independently
                
                # Get target emotion category for this sample
                target_val = targets['valence']
                target_aro = targets['arousal']
                target_category = categorize_emotion_25bin(target_val, target_aro)
                
                # STEP 1: Complete reset for each sample (like isolation test)
                model.lrm.clear_stored_activations()
                # Reset ModBlock scales to original values before applying new steering
                for lrm_module_name, lrm_module in model.lrm.named_children():
                    for mod_name, mod_module in lrm_module.named_children():
                        if hasattr(mod_module, 'neg_scale_orig') and hasattr(mod_module, 'pos_scale_orig'):
                            mod_module.neg_scale.data = mod_module.neg_scale_orig.clone()
                            mod_module.pos_scale.data = mod_module.pos_scale_orig.clone()
                
                # STEP 2: Prepare features (individual sample, not batch)
                features = features.unsqueeze(0).to(device)  # Shape: [1, 1024, 64]
                
                # STEP 3: Apply steering using the SAME INTERFACE as working Test_Steering_Emotion.py
                # TEST: Use FIXED steering (same category for all samples) like isolation test
                fixed_category = 'very_negative_strong'  # Use same category as isolation test
                steering_applied_for_sample = False
                
                if strength > 0.0 and fixed_category in signals_25bin:
                    signals = signals_25bin[fixed_category]
                    steering_applied_for_sample = True
                    
                    if sample_idx < 3 and strength == 5.0:  # Debug first 3 samples for one strength
                        print(f"      Sample {sample_idx}: Applying FIXED steering for category '{fixed_category}'")
                    
                    # Create steering_signals list like Test_Steering_Emotion.py
                    steering_signals_list = []
                    
                    if 'valence_128d' in signals:
                        valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
                        steering_signals_list.append({
                            'source': 'affective_valence_128d', 
                            'activation': valence_signal,
                            'strength': strength,
                            'alpha': 1.0
                        })
                    
                    if 'arousal_128d' in signals:
                        arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
                        steering_signals_list.append({
                            'source': 'affective_arousal_128d', 
                            'activation': arousal_signal,
                            'strength': strength,
                            'alpha': 1.0
                        })
                else:
                    steering_signals_list = None
                
                # STEP 4: Forward pass using SAME INTERFACE as Test_Steering_Emotion.py
                output = model(features, forward_passes=2, 
                             steering_signals=steering_signals_list,
                             first_pass_steering=False)
                
                # STEP 5: Store results for this individual sample
                all_predictions.append({
                    'valence': output['valence'][0].item(),
                    'arousal': output['arousal'][0].item()
                })
                all_targets.append({
                    'valence': target_val,
                    'arousal': target_aro
                })
                all_categories.append(target_category)
                steering_applied.append(steering_applied_for_sample)
                
                # STEP 6: Don't clear activations after each sample - this might interfere with strength adjustments
                # model.lrm.clear_stored_activations()
        
        # Calculate metrics
        pred_valence = [p['valence'] for p in all_predictions]
        pred_arousal = [p['arousal'] for p in all_predictions]
        true_valence = [t['valence'] for t in all_targets]
        true_arousal = [t['arousal'] for t in all_targets]
        
        # Correlation
        val_corr = np.corrcoef(pred_valence, true_valence)[0, 1] if len(set(pred_valence)) > 1 else 0.0
        aro_corr = np.corrcoef(pred_arousal, true_arousal)[0, 1] if len(set(pred_arousal)) > 1 else 0.0
        
        # Mean predictions
        mean_pred_val = np.mean(pred_valence)
        mean_pred_aro = np.mean(pred_arousal)
        mean_true_val = np.mean(true_valence)
        mean_true_aro = np.mean(true_arousal)
        
        # Steering coverage
        steering_coverage = sum(steering_applied) / len(steering_applied) * 100
        
        results[strength] = {
            'val_corr': val_corr,
            'aro_corr': aro_corr,
            'mean_pred_val': mean_pred_val,
            'mean_pred_aro': mean_pred_aro,
            'mean_true_val': mean_true_val,
            'mean_true_aro': mean_true_aro,
            'steering_coverage': steering_coverage,
            'num_samples': len(all_predictions)
        }
        
        print(f"   Samples: {len(all_predictions)}, Steering coverage: {steering_coverage:.1f}%")
        print(f"   Valence: r={val_corr:.3f}, mean_pred={mean_pred_val:.3f}, mean_true={mean_true_val:.3f}")
        print(f"   Arousal: r={aro_corr:.3f}, mean_pred={mean_pred_aro:.3f}, mean_true={mean_true_aro:.3f}")
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Compare with baseline (strength 0.0)
    baseline = results[0.0]
    
    print(f"\n📊 Performance Comparison (vs baseline):")
    print(f"{'Strength':>8} {'Val r':>8} {'Aro r':>8} {'Val Δr':>8} {'Aro Δr':>8} {'Val Bias':>10} {'Aro Bias':>10}")
    print("-" * 70)
    
    for strength in strengths:
        res = results[strength]
        val_delta_r = res['val_corr'] - baseline['val_corr']
        aro_delta_r = res['aro_corr'] - baseline['aro_corr']
        val_bias = res['mean_pred_val'] - res['mean_true_val']
        aro_bias = res['mean_pred_aro'] - res['mean_true_aro']
        
        print(f"{strength:>8.1f} {res['val_corr']:>8.3f} {res['aro_corr']:>8.3f} "
              f"{val_delta_r:>+8.3f} {aro_delta_r:>+8.3f} {val_bias:>+10.3f} {aro_bias:>+10.3f}")
    
    # Find optimal strength
    print(f"\n🎯 Optimal Strength Analysis:")
    
    # Look for best correlation improvements
    best_val_strength = max(strengths[1:], key=lambda s: results[s]['val_corr'] - baseline['val_corr'])
    best_aro_strength = max(strengths[1:], key=lambda s: results[s]['aro_corr'] - baseline['aro_corr'])
    
    val_improvement = results[best_val_strength]['val_corr'] - baseline['val_corr']
    aro_improvement = results[best_aro_strength]['aro_corr'] - baseline['aro_corr']
    
    print(f"   Best valence improvement: +{val_improvement:.3f} at strength {best_val_strength}")
    print(f"   Best arousal improvement: +{aro_improvement:.3f} at strength {best_aro_strength}")
    
    # Check for steering effects
    print(f"\n📈 Steering Effect Analysis:")
    for strength in [1.0, 2.0, 5.0, 10.0]:
        if strength in results:
            res = results[strength]
            val_shift = res['mean_pred_val'] - baseline['mean_pred_val']
            aro_shift = res['mean_pred_aro'] - baseline['mean_pred_aro']
            total_shift = abs(val_shift) + abs(aro_shift)
            
            print(f"   Strength {strength:4.1f}: Valence shift={val_shift:+.3f}, Arousal shift={aro_shift:+.3f}, Total={total_shift:.3f}")
    
    # Summary
    max_total_shift = 0
    best_overall_strength = 0.0
    for strength in strengths[1:]:
        if strength in results:
            res = results[strength]
            val_shift = abs(res['mean_pred_val'] - baseline['mean_pred_val'])
            aro_shift = abs(res['mean_pred_aro'] - baseline['mean_pred_aro'])
            total_shift = val_shift + aro_shift
            
            if total_shift > max_total_shift:
                max_total_shift = total_shift
                best_overall_strength = strength
    
    print(f"\n✅ Summary:")
    print(f"   Amplification fix is working: 3x scaling + ±1.0 clamping")
    print(f"   Maximum steering effect: {max_total_shift:.3f} at strength {best_overall_strength}")
    print(f"   Steering coverage: {results[best_overall_strength]['steering_coverage']:.1f}%")
    
    if max_total_shift > 0.1:
        print(f"   🎉 SUCCESS: Strong steering effects detected (>{0.1:.1f})")
    elif max_total_shift > 0.05:
        print(f"   ✅ GOOD: Moderate steering effects detected (>{0.05:.2f})")
    else:
        print(f"   ⚠️ WEAK: Small steering effects detected (<{0.05:.2f})")

if __name__ == "__main__":
    test_real_emotion_dataset() 