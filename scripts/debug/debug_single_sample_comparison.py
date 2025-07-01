#!/usr/bin/env python3
"""
Debug script to test all steering methods on a single sample.
This will help isolate why the comprehensive test shows identical results.
"""

import os
import sys
import torch
import numpy as np
import json
import h5py

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def load_emotion_model(checkpoint_path, device):
    """Load the emotion model with LRM capabilities."""
    print(f"📂 Loading model from: {checkpoint_path}")
    
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        freeze_base=True
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model

def categorize_9bin(valence, arousal):
    """9-bin categorization."""
    v_cat = 'negative' if valence < -0.33 else ('positive' if valence > 0.33 else 'neutral')
    a_cat = 'weak' if arousal < -0.33 else ('strong' if arousal > 0.33 else 'middle')
    return f"{v_cat}_{a_cat}"

def categorize_25bin_with_fallback(valence, arousal):
    """25-bin categorization with fallback."""
    v_bins = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
    a_bins = ['very_weak', 'weak', 'middle', 'strong', 'very_strong']
    
    v_idx = max(0, min(4, int((valence + 1) * 2.5)))
    a_idx = max(0, min(4, int((arousal + 1) * 2.5)))
    
    category = f"{v_bins[v_idx]}_{a_bins[a_idx]}"
    
    # Smart fallback for missing categories
    fallback_mapping = {
        'very_positive_very_strong': 'very_positive_strong',
        'negative_very_weak': 'neutral_very_weak'
    }
    
    if category in fallback_mapping:
        category = fallback_mapping[category]
    
    return category

def test_single_sample_all_methods():
    """Test all steering methods on a single sample."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load model
    checkpoint_path = 'workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    model = load_emotion_model(checkpoint_path, device)
    
    # Load dataset
    dataset_path = 'workspaces/emotion_feedback/features/emotion_features.h5'
    with h5py.File(dataset_path, 'r') as hf:
        features = hf['feature'][:]
        valence_targets = hf['valence'][:]
        arousal_targets = hf['arousal'][:]
    
    # Use validation split
    np.random.seed(42)
    total_samples = len(features)
    indices = np.random.permutation(total_samples)
    train_size = int(total_samples * 0.7)
    val_indices = indices[train_size:]
    
    features = features[val_indices]
    valence_targets = valence_targets[val_indices]
    arousal_targets = arousal_targets[val_indices]
    
    # Load steering signals
    with open('tmp/steering_signals_by_category.json', 'r') as f:
        steering_signals_9bin = json.load(f)
    with open('tmp/25bin_steering_signals/steering_signals_25bin.json', 'r') as f:
        steering_signals_25bin = json.load(f)
    
    # Test first sample
    sample_idx = 0
    sample_tensor = torch.tensor(features[sample_idx:sample_idx+1], dtype=torch.float32).to(device)
    target_v = valence_targets[sample_idx]
    target_a = arousal_targets[sample_idx]
    
    print(f"\n🧪 TESTING SAMPLE {sample_idx}")
    print(f"📊 Target: V={target_v:.3f}, A={target_a:.3f}")
    
    results = {}
    
    # 1. Baseline (no steering)
    print(f"\n1️⃣ BASELINE TEST")
    model.clear_feedback_state()
    with torch.no_grad():
        baseline_output = model(sample_tensor)
    baseline_v = baseline_output['valence'].cpu().item()
    baseline_a = baseline_output['arousal'].cpu().item()
    results['baseline'] = {'valence': baseline_v, 'arousal': baseline_a}
    print(f"🎯 Baseline: V={baseline_v:.6f}, A={baseline_a:.6f}")
    
    # 2. 9-bin categorical steering
    print(f"\n2️⃣ 9-BIN CATEGORICAL TEST")
    category_9bin = categorize_9bin(target_v, target_a)
    print(f"🔍 Category: {category_9bin}")
    
    if category_9bin in steering_signals_9bin:
        signals = steering_signals_9bin[category_9bin]
        
        # Clear state and apply steering
        model.clear_feedback_state()
        
        if 'valence_128d' in signals:
            model.add_steering_signal(
                source='affective_valence_128d',
                activation=torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device),
                strength=5.0,
                alpha=1.0
            )
        
        if 'arousal_128d' in signals:
            model.add_steering_signal(
                source='affective_arousal_128d',
                activation=torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device),
                strength=5.0,
                alpha=1.0
            )
        
        model.lrm.enable()
        
        with torch.no_grad():
            steered_output = model(sample_tensor, forward_passes=2)
        steered_v = steered_output['valence'].cpu().item()
        steered_a = steered_output['arousal'].cpu().item()
        results['9bin'] = {'valence': steered_v, 'arousal': steered_a}
        
        print(f"🎯 9-bin steered: V={steered_v:.6f}, A={steered_a:.6f}")
        print(f"📈 9-bin effect: ΔV={steered_v-baseline_v:+.6f}, ΔA={steered_a-baseline_a:+.6f}")
    else:
        print(f"❌ Category not found")
        results['9bin'] = results['baseline']
    
    # 3. 25-bin categorical steering
    print(f"\n3️⃣ 25-BIN CATEGORICAL TEST")
    category_25bin = categorize_25bin_with_fallback(target_v, target_a)
    print(f"🔍 Category: {category_25bin}")
    
    if category_25bin in steering_signals_25bin:
        signals = steering_signals_25bin[category_25bin]
        
        # Clear state and apply steering
        model.clear_feedback_state()
        
        if 'valence_128d' in signals:
            model.add_steering_signal(
                source='affective_valence_128d',
                activation=torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device),
                strength=5.0,
                alpha=1.0
            )
        
        if 'arousal_128d' in signals:
            model.add_steering_signal(
                source='affective_arousal_128d',
                activation=torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device),
                strength=5.0,
                alpha=1.0
            )
        
        model.lrm.enable()
        
        with torch.no_grad():
            steered_output = model(sample_tensor, forward_passes=2)
        steered_v = steered_output['valence'].cpu().item()
        steered_a = steered_output['arousal'].cpu().item()
        results['25bin'] = {'valence': steered_v, 'arousal': steered_a}
        
        print(f"🎯 25-bin steered: V={steered_v:.6f}, A={steered_a:.6f}")
        print(f"📈 25-bin effect: ΔV={steered_v-baseline_v:+.6f}, ΔA={steered_a-baseline_a:+.6f}")
    else:
        print(f"❌ Category not found")
        results['25bin'] = results['baseline']
    
    # 4. Summary comparison
    print(f"\n📊 SUMMARY COMPARISON")
    print(f"{'Method':<15} {'Valence':<12} {'Arousal':<12} {'ΔV':<12} {'ΔA':<12}")
    print("-" * 65)
    
    baseline_v = results['baseline']['valence']
    baseline_a = results['baseline']['arousal']
    
    for method_name, method_label in [('baseline', 'Baseline'), ('9bin', '9-bin'), ('25bin', '25-bin')]:
        if method_name in results:
            v = results[method_name]['valence']
            a = results[method_name]['arousal']
            dv = v - baseline_v if method_name != 'baseline' else 0.0
            da = a - baseline_a if method_name != 'baseline' else 0.0
            
            print(f"{method_label:<15} {v:<12.6f} {a:<12.6f} {dv:<+12.6f} {da:<+12.6f}")
    
    # Check if all methods produce identical results
    if len(set(results[k]['valence'] for k in results)) == 1:
        print(f"\n⚠️  WARNING: All methods produce IDENTICAL valence predictions!")
    if len(set(results[k]['arousal'] for k in results)) == 1:
        print(f"⚠️  WARNING: All methods produce IDENTICAL arousal predictions!")
    
    if results['9bin']['valence'] != results['baseline']['valence'] or results['9bin']['arousal'] != results['baseline']['arousal']:
        print(f"✅ 9-bin steering shows effect!")
    else:
        print(f"❌ 9-bin steering shows NO effect!")
        
    if results['25bin']['valence'] != results['baseline']['valence'] or results['25bin']['arousal'] != results['baseline']['arousal']:
        print(f"✅ 25-bin steering shows effect!")
    else:
        print(f"❌ 25-bin steering shows NO effect!")

if __name__ == "__main__":
    test_single_sample_all_methods() 