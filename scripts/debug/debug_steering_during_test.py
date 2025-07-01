#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

import torch
import numpy as np
import json

from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def test_single_steering():
    """Test steering signals on a single sample with debug output."""
    print("=== Testing Single Sample Steering ===")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000, window_size=1024, hop_size=320, 
        mel_bins=64, fmin=50, fmax=14000, forward_passes=2
    )
    model.load_from_pretrain('/DATA/pliu/EmotionData/Cnn6_mAP=0.343.pth')
    model = model.to(device)
    model.eval()
    
    # Load steering signals
    with open('steering_signals_25bin/steering_signals_25bin.json', 'r') as f:
        steering_data = json.load(f)
    
    # Create a dummy input
    dummy_input = torch.randn(1, 1024, 64).to(device)
    
    print("\n--- Baseline Forward Pass ---")
    with torch.no_grad():
        baseline_output = model(dummy_input, forward_passes=2)
        baseline_valence = baseline_output['valence'].item()
        baseline_arousal = baseline_output['arousal'].item()
        print(f"Baseline - Valence: {baseline_valence:.4f}, Arousal: {baseline_arousal:.4f}")
    
    print("\n--- Steering Forward Pass ---")
    # Get steering signals
    category = 'positive_strong'  # Should increase both valence and arousal
    if category in steering_data:
        valence_signal = torch.tensor(steering_data[category]['valence_128d'], dtype=torch.float32).to(device)
        arousal_signal = torch.tensor(steering_data[category]['arousal_128d'], dtype=torch.float32).to(device)
        
        steering_signals_list = [
            {'source': 'affective_valence_128d', 'activation': valence_signal, 'strength': 5.0, 'alpha': 1.0},
            {'source': 'affective_arousal_128d', 'activation': arousal_signal, 'strength': 5.0, 'alpha': 1.0}
        ]
        
        print(f"Steering signals prepared for category: {category}")
        print(f"Valence signal shape: {valence_signal.shape}")
        print(f"Arousal signal shape: {arousal_signal.shape}")
        
        with torch.no_grad():
            model.lrm.enable()
            
            # Add debug output before steering
            print(f"\n--- Debug: LRM Module Structure ---")
            for lrm_name, lrm_module in model.lrm.named_children():
                print(f"LRM Module: {lrm_name}")
                for mod_name, mod_module in lrm_module.named_children():
                    print(f"  ModBlock: {mod_name}")
            
            print(f"\n--- Debug: Calling model forward with steering ---")
            print(f"Steering signals list: {[s['source'] for s in steering_signals_list]}")
            
            steered_output = model(dummy_input, forward_passes=2, steering_signals=steering_signals_list, first_pass_steering=True)
            steered_valence = steered_output['valence'].item()
            steered_arousal = steered_output['arousal'].item()
            
            print(f"Steered - Valence: {steered_valence:.4f}, Arousal: {steered_arousal:.4f}")
            print(f"Changes - Valence: {steered_valence - baseline_valence:+.4f}, Arousal: {steered_arousal - baseline_arousal:+.4f}")
            
            # Check if LRM modules have stored activations
            print("\n--- LRM Module States ---")
            for lrm_name, lrm_module in model.lrm.named_children():
                print(f"LRM Module: {lrm_name}")
                print(f"  mod_inputs keys: {list(lrm_module.mod_inputs.keys())}")
                print(f"  active_connections: {lrm_module.active_connections}")
                if hasattr(lrm_module, 'pre_mod_output') and lrm_module.pre_mod_output is not None:
                    print(f"  pre_mod_output shape: {lrm_module.pre_mod_output.shape}")
                if hasattr(lrm_module, 'total_mod') and lrm_module.total_mod is not None:
                    print(f"  total_mod shape: {lrm_module.total_mod.shape}")
                    print(f"  total_mod mean: {lrm_module.total_mod.mean().item():.6f}")
                    print(f"  total_mod std: {lrm_module.total_mod.std().item():.6f}")
                if hasattr(lrm_module, 'post_mod_output') and lrm_module.post_mod_output is not None:
                    print(f"  post_mod_output shape: {lrm_module.post_mod_output.shape}")
                print()
    
    else:
        print(f"Category {category} not found in steering signals")

if __name__ == "__main__":
    test_single_steering() 