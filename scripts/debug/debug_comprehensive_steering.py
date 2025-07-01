#!/usr/bin/env python3
"""
Debug script to investigate why all steering methods produce identical results.
"""

import os
import sys
import torch
import numpy as np
import json
import h5py
from pathlib import Path

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

def test_single_sample_steering():
    """Test steering on a single sample with detailed debugging."""
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
        audio_names = [name.decode('utf-8') for name in hf['audio_name'][:]]
    
    # Use validation split (same as comprehensive test)
    np.random.seed(42)
    total_samples = len(features)
    indices = np.random.permutation(total_samples)
    train_size = int(total_samples * 0.7)
    val_indices = indices[train_size:]
    
    features = features[val_indices][:5]  # Just first 5 validation samples
    valence_targets = valence_targets[val_indices][:5]
    arousal_targets = arousal_targets[val_indices][:5]
    
    # Load steering signals
    steering_9bin_path = 'tmp/steering_signals_by_category.json'
    steering_25bin_path = 'tmp/25bin_steering_signals/steering_signals_25bin.json'
    
    with open(steering_9bin_path, 'r') as f:
        steering_signals_9bin = json.load(f)
    with open(steering_25bin_path, 'r') as f:
        steering_signals_25bin = json.load(f)
    
    print(f"📊 Testing with {len(features)} samples")
    print(f"🎯 Available 9-bin categories: {len(steering_signals_9bin)}")
    print(f"🎯 Available 25-bin categories: {len(steering_signals_25bin)}")
    
    # Test first sample
    sample_idx = 0
    sample_tensor = torch.tensor(features[sample_idx:sample_idx+1], dtype=torch.float32).to(device)
    target_v = valence_targets[sample_idx]
    target_a = arousal_targets[sample_idx]
    
    print(f"\n🧪 TESTING SAMPLE {sample_idx}")
    print(f"📊 Target: V={target_v:.3f}, A={target_a:.3f}")
    
    # 1. Baseline prediction
    model.clear_feedback_state()
    with torch.no_grad():
        baseline_output = model(sample_tensor)
    baseline_v = baseline_output['valence'].cpu().item()
    baseline_a = baseline_output['arousal'].cpu().item()
    print(f"🎯 Baseline: V={baseline_v:.3f}, A={baseline_a:.3f}")
    
    # 2. Test 9-bin steering
    def categorize_9bin(valence, arousal):
        v_cat = 'negative' if valence < -0.33 else ('positive' if valence > 0.33 else 'neutral')
        a_cat = 'weak' if arousal < -0.33 else ('strong' if arousal > 0.33 else 'middle')
        return f"{v_cat}_{a_cat}"
    
    category_9bin = categorize_9bin(target_v, target_a)
    print(f"🔍 9-bin category: {category_9bin}")
    
    if category_9bin in steering_signals_9bin:
        signals = steering_signals_9bin[category_9bin]
        
        model.clear_feedback_state()
        
        if 'valence_128d' in signals:
            v_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
            print(f"📡 Valence signal: shape={v_signal.shape}, mean={v_signal.mean():.6f}")
            model.add_steering_signal(
                source='affective_valence_128d',
                activation=v_signal,
                strength=5.0,
                alpha=1.0
            )
        
        if 'arousal_128d' in signals:
            a_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
            print(f"📡 Arousal signal: shape={a_signal.shape}, mean={a_signal.mean():.6f}")
            model.add_steering_signal(
                source='affective_arousal_128d',
                activation=a_signal,
                strength=5.0,
                alpha=1.0
            )
        
        model.lrm.enable()
        print(f"📊 LRM enabled: {model.lrm.enabled}")
        print(f"📊 LRM mod_inputs keys: {list(model.lrm.mod_inputs.keys())}")
        
        with torch.no_grad():
            steered_output = model(sample_tensor, forward_passes=2)
        steered_v = steered_output['valence'].cpu().item()
        steered_a = steered_output['arousal'].cpu().item()
        
        print(f"🎯 9-bin steered: V={steered_v:.3f}, A={steered_a:.3f}")
        print(f"📈 9-bin effect: ΔV={steered_v-baseline_v:+.6f}, ΔA={steered_a-baseline_a:+.6f}")
        
        if abs(steered_v - baseline_v) < 1e-6 and abs(steered_a - baseline_a) < 1e-6:
            print("⚠️  WARNING: No steering effect detected!")
        else:
            print("✅ Steering effect confirmed!")
    else:
        print(f"❌ Category {category_9bin} not found in 9-bin signals")
    
    # 3. Test 25-bin steering with same sample
    def categorize_25bin_with_fallback(valence, arousal):
        # 25-bin categorization logic
        v_bins = ['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
        a_bins = ['very_weak', 'weak', 'middle', 'strong', 'very_strong']
        
        v_idx = max(0, min(4, int((valence + 1) * 2.5)))
        a_idx = max(0, min(4, int((arousal + 1) * 2.5)))
        
        return f"{v_bins[v_idx]}_{a_bins[a_idx]}"
    
    category_25bin = categorize_25bin_with_fallback(target_v, target_a)
    print(f"\n🔍 25-bin category: {category_25bin}")
    
    if category_25bin in steering_signals_25bin:
        signals = steering_signals_25bin[category_25bin]
        
        model.clear_feedback_state()
        
        if 'valence_128d' in signals:
            v_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
            print(f"📡 Valence signal: shape={v_signal.shape}, mean={v_signal.mean():.6f}")
            model.add_steering_signal(
                source='affective_valence_128d',
                activation=v_signal,
                strength=5.0,
                alpha=1.0
            )
        
        if 'arousal_128d' in signals:
            a_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
            print(f"📡 Arousal signal: shape={a_signal.shape}, mean={a_signal.mean():.6f}")
            model.add_steering_signal(
                source='affective_arousal_128d',
                activation=a_signal,
                strength=5.0,
                alpha=1.0
            )
        
        model.lrm.enable()
        
        with torch.no_grad():
            steered_output = model(sample_tensor, forward_passes=2)
        steered_v = steered_output['valence'].cpu().item()
        steered_a = steered_output['arousal'].cpu().item()
        
        print(f"🎯 25-bin steered: V={steered_v:.3f}, A={steered_a:.3f}")
        print(f"📈 25-bin effect: ΔV={steered_v-baseline_v:+.6f}, ΔA={steered_a-baseline_a:+.6f}")
        
        if abs(steered_v - baseline_v) < 1e-6 and abs(steered_a - baseline_a) < 1e-6:
            print("⚠️  WARNING: No steering effect detected!")
        else:
            print("✅ Steering effect confirmed!")
    else:
        print(f"❌ Category {category_25bin} not found in 25-bin signals")

if __name__ == "__main__":
    test_single_sample_steering() 