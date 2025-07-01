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

def test_simplified():
    """Simplified test matching the working debug script."""
    
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
    
    # Create test samples
    print("🔧 Creating test samples...")
    np.random.seed(42)
    
    test_samples = []
    for i in range(10):  # Just 10 samples for quick testing
        features = torch.randn(1, 1024, 64).to(device)
        valence = np.random.uniform(-0.8, 0.8)
        arousal = np.random.uniform(-0.8, 0.8)
        category = categorize_emotion_25bin(valence, arousal)
        test_samples.append((features, valence, arousal, category))
    
    print(f"✅ Created {len(test_samples)} test samples")
    
    print("\n" + "="*60)
    print("SIMPLIFIED STEERING TEST")
    print("="*60)
    
    # Test baseline vs steering
    strengths = [0.0, 5.0]
    all_predictions = {}  # Store predictions by strength
    
    for strength in strengths:
        print(f"\n🔍 Testing strength: {strength}")
        
        predictions = []
        steering_applied_count = 0
        
        for i, (features, true_val, true_aro, category) in enumerate(test_samples):
            
            # Clear previous state
            model.lrm.clear_stored_activations()
            
            if strength > 0.0 and category in signals_25bin:
                # Apply steering using the same method as debug script
                signals = signals_25bin[category]
                
                if 'valence_128d' in signals:
                    valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
                    model.add_steering_signal('affective_valence_128d', valence_signal, strength=strength)
                
                if 'arousal_128d' in signals:
                    arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
                    model.add_steering_signal('affective_arousal_128d', arousal_signal, strength=strength)
                
                model.lrm.enable()
                steering_applied_count += 1
            else:
                model.lrm.disable()
            
            # Forward pass
            with torch.no_grad():
                output = model(features, forward_passes=2)
            
            pred_val = output['valence'][0].item()
            pred_aro = output['arousal'][0].item()
            predictions.append((pred_val, pred_aro))
            
            if i < 3:  # Show details for first 3 samples
                print(f"   Sample {i}: category={category}")
                print(f"     True: val={true_val:.3f}, aro={true_aro:.3f}")
                print(f"     Pred: val={pred_val:.3f}, aro={pred_aro:.3f}")
                if strength > 0.0 and category in signals_25bin:
                    print(f"     ✅ Steering applied")
                else:
                    print(f"     ❌ No steering (category not found or strength=0)")
        
        # Calculate stats
        pred_vals = [p[0] for p in predictions]
        pred_aros = [p[1] for p in predictions]
        
        mean_val = np.mean(pred_vals)
        mean_aro = np.mean(pred_aros)
        std_val = np.std(pred_vals)
        std_aro = np.std(pred_aros)
        
        coverage = steering_applied_count / len(test_samples) * 100
        
        print(f"   Results: {len(predictions)} samples, {coverage:.1f}% steering coverage")
        print(f"   Valence: mean={mean_val:.4f}, std={std_val:.4f}")
        print(f"   Arousal: mean={mean_aro:.4f}, std={std_aro:.4f}")
        
        # Store results for comparison
        all_predictions[strength] = {
            'predictions': predictions,
            'coverage': coverage,
            'mean_val': mean_val,
            'mean_aro': mean_aro
        }
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    # Compare baseline vs steering
    if 0.0 in all_predictions and 5.0 in all_predictions:
        baseline = all_predictions[0.0]
        steering = all_predictions[5.0]
        
        val_shift = steering['mean_val'] - baseline['mean_val']
        aro_shift = steering['mean_aro'] - baseline['mean_aro']
        
        print(f"Baseline (strength 0.0): val={baseline['mean_val']:.4f}, aro={baseline['mean_aro']:.4f}")
        print(f"Steering (strength 5.0): val={steering['mean_val']:.4f}, aro={steering['mean_aro']:.4f}")
        print(f"Changes:                 val={val_shift:+.4f}, aro={aro_shift:+.4f}")
        print(f"Steering coverage:       {steering['coverage']:.1f}%")
        
        if abs(val_shift) > 0.01 or abs(aro_shift) > 0.01:
            print("✅ SUCCESS: Steering effects detected!")
        else:
            print("❌ FAILED: No steering effects detected")
            print("   This suggests steering signals are stored but not affecting predictions")
    else:
        print("❌ ERROR: Missing baseline or steering results")

if __name__ == "__main__":
    test_simplified() 