#!/usr/bin/env python3

import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import json
import numpy as np
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
import h5py

def test_dual_signal_processing():
    """Test that valence and arousal steering signals are processed independently."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔬 DUAL SIGNAL PROCESSING TEST")
    print(f"Using device: {device}")
    
    # Load model
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, 
        fmin=50, fmax=14000, forward_passes=2
    ).to(device)
    
    checkpoint_path = '/home/pengliu/Private/my_panns_transfer_to_gtzan/workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        print("✅ Loaded model")
    else:
        print(f"❌ Error: Model not found")
        return
    
    # Load steering signals
    steering_signals_path = './steering_signals_25bin/steering_signals_25bin.json'
    if not os.path.exists(steering_signals_path):
        steering_signals_path = './tmp/25bin_steering_signals/steering_signals_25bin.json'
    
    with open(steering_signals_path, 'r') as f:
        signals_25bin = json.load(f)
    
    # Load test sample
    dataset_paths = ['workspaces/emotion_feedback/features/emotion_features.h5']
    test_sample = None
    for path in dataset_paths:
        if os.path.exists(path):
            with h5py.File(path, 'r') as hf:
                test_sample = torch.tensor(hf['feature'][0], dtype=torch.float32).unsqueeze(0).to(device)
            break
    
    if test_sample is None:
        print("❌ Error: Could not load test sample")
        return
    
    print(f"✅ Loaded test sample: {test_sample.shape}")
    
    # Get baseline prediction
    with torch.no_grad():
        baseline = model(test_sample, forward_passes=2, steering_signals=None)
        baseline_val = baseline['valence'][0].item()
        baseline_aro = baseline['arousal'][0].item()
    
    print(f"\n📊 Baseline: Valence={baseline_val:.4f}, Arousal={baseline_aro:.4f}")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Valence Only (Positive)',
            'category': 'very_positive_moderate',
            'use_valence': True,
            'use_arousal': False
        },
        {
            'name': 'Arousal Only (High)',
            'category': 'neutral_very_strong', 
            'use_valence': False,
            'use_arousal': True
        },
        {
            'name': 'Both Signals (Pos+High)',
            'category': 'very_positive_very_strong',
            'use_valence': True,
            'use_arousal': True
        },
        {
            'name': 'Valence Only (Negative)',
            'category': 'very_negative_moderate',
            'use_valence': True,
            'use_arousal': False
        },
        {
            'name': 'Arousal Only (Low)',
            'category': 'neutral_very_weak',
            'use_valence': False,
            'use_arousal': True
        },
        {
            'name': 'Both Signals (Neg+Low)',
            'category': 'very_negative_very_weak',
            'use_valence': True,
            'use_arousal': True
        }
    ]
    
    print(f"\n🧪 INDEPENDENCE TEST")
    print(f"Testing if valence and arousal signals work independently...")
    print(f"{'Config':<25} {'Val Pred':<10} {'Aro Pred':<10} {'Val Δ':<10} {'Aro Δ':<10} {'Signals':<15}")
    print("-" * 85)
    
    strength = 1.0
    results = []
    
    for config in test_configs:
        category = config['category']
        
        # Check if category exists
        if category not in signals_25bin:
            print(f"   Skipping {config['name']} - category not found")
            continue
        
        signals = signals_25bin[category]
        steering_signals_current = []
        
        # Add valence signal if requested
        if config['use_valence'] and 'valence_128d' in signals:
            valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
            steering_signals_current.append({
                'source': 'affective_valence_128d',
                'activation': valence_signal,
                'strength': strength,
                'alpha': 1.0
            })
        
        # Add arousal signal if requested
        if config['use_arousal'] and 'arousal_128d' in signals:
            arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
            steering_signals_current.append({
                'source': 'affective_arousal_128d',
                'activation': arousal_signal,
                'strength': strength,
                'alpha': 1.0
            })
        
        # Forward pass
        with torch.no_grad():
            output = model(test_sample, 
                         forward_passes=2,
                         steering_signals=steering_signals_current,
                         first_pass_steering=False)
        
        pred_val = output['valence'][0].item()
        pred_aro = output['arousal'][0].item()
        
        val_change = pred_val - baseline_val
        aro_change = pred_aro - baseline_aro
        
        # Create signal indicator
        signals_used = []
        if config['use_valence']:
            signals_used.append('V')
        if config['use_arousal']:
            signals_used.append('A')
        signals_str = '+'.join(signals_used) if signals_used else 'None'
        
        results.append({
            'name': config['name'],
            'pred_val': pred_val,
            'pred_aro': pred_aro,
            'val_change': val_change,
            'aro_change': aro_change,
            'use_valence': config['use_valence'],
            'use_arousal': config['use_arousal']
        })
        
        print(f"{config['name']:<25} {pred_val:<10.4f} {pred_aro:<10.4f} {val_change:<+10.4f} {aro_change:<+10.4f} {signals_str:<15}")
    
    # Analyze independence
    print(f"\n🔍 INDEPENDENCE ANALYSIS")
    
    # Find valence-only and arousal-only results
    val_only_results = [r for r in results if r['use_valence'] and not r['use_arousal']]
    aro_only_results = [r for r in results if not r['use_valence'] and r['use_arousal']]
    both_results = [r for r in results if r['use_valence'] and r['use_arousal']]
    
    if val_only_results:
        avg_val_change_from_val = np.mean([r['val_change'] for r in val_only_results])
        avg_aro_change_from_val = np.mean([r['aro_change'] for r in val_only_results])
        print(f"Valence-only steering: ΔV={avg_val_change_from_val:+.4f}, ΔA={avg_aro_change_from_val:+.4f}")
    
    if aro_only_results:
        avg_val_change_from_aro = np.mean([r['val_change'] for r in aro_only_results])
        avg_aro_change_from_aro = np.mean([r['aro_change'] for r in aro_only_results])
        print(f"Arousal-only steering: ΔV={avg_val_change_from_aro:+.4f}, ΔA={avg_aro_change_from_aro:+.4f}")
    
    # Check for independence
    print(f"\n📊 INDEPENDENCE CHECK:")
    
    val_affects_val = val_only_results and np.mean([abs(r['val_change']) for r in val_only_results]) > 0.01
    val_affects_aro = val_only_results and np.mean([abs(r['aro_change']) for r in val_only_results]) > 0.01
    aro_affects_val = aro_only_results and np.mean([abs(r['val_change']) for r in aro_only_results]) > 0.01
    aro_affects_aro = aro_only_results and np.mean([abs(r['aro_change']) for r in aro_only_results]) > 0.01
    
    print(f"   Valence signals affect valence predictions: {'✓' if val_affects_val else '✗'}")
    print(f"   Valence signals affect arousal predictions: {'✗' if not val_affects_aro else '⚠️'}")
    print(f"   Arousal signals affect valence predictions: {'✗' if not aro_affects_val else '⚠️'}")
    print(f"   Arousal signals affect arousal predictions: {'✓' if aro_affects_aro else '✗'}")
    
    # Check for proper signal strength
    print(f"\n🎯 SIGNAL STRENGTH CHECK:")
    if val_only_results:
        max_val_effect = max([abs(r['val_change']) for r in val_only_results])
        print(f"   Maximum valence effect: {max_val_effect:.4f}")
    
    if aro_only_results:
        max_aro_effect = max([abs(r['aro_change']) for r in aro_only_results])
        print(f"   Maximum arousal effect: {max_aro_effect:.4f}")
    
    # Final assessment
    print(f"\n🏁 FINAL ASSESSMENT:")
    
    if val_affects_val and aro_affects_aro:
        print("   ✅ Both valence and arousal signals are working")
    else:
        print("   ❌ Some signals are not working properly")
    
    if not val_affects_aro and not aro_affects_val:
        print("   ✅ Signals are properly isolated to their respective dimensions")
    else:
        print("   ⚠️  Cross-dimensional interference detected")
    
    if (val_only_results and max([abs(r['val_change']) for r in val_only_results]) > 0.05) or \
       (aro_only_results and max([abs(r['aro_change']) for r in aro_only_results]) > 0.05):
        print("   ✅ Signal strength is sufficient")
    else:
        print("   ⚠️  Signal strength may be too weak")

if __name__ == "__main__":
    test_dual_signal_processing() 