#!/usr/bin/env python3
"""
Detailed debug script to investigate LRM connections and feedback signal application.
This script will help identify exactly why the feedback mechanism isn't working.
"""

import os
import sys
import torch
import numpy as np
import argparse
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
from data.data_generator import EmoSoundscapesDataset, EmotionValidateSampler, emotion_collate_fn
from configs.model_configs import cnn6_config

def debug_lrm_connections(model_path, dataset_path, num_samples=1):
    """Debug the LRM connections and feedback signal application."""
    print("🔍 Debugging LRM Connections and Feedback Signals")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\n📦 Loading model...")
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000,
        window_size=cnn6_config['window_size'],
        hop_size=cnn6_config['hop_size'],
        mel_bins=cnn6_config['mel_bins'],
        fmin=cnn6_config['fmin'],
        fmax=cnn6_config['fmax'],
        freeze_base=True,
        forward_passes=2
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {model_path}")
    
    # Check LRM structure
    print(f"\n🔍 Checking LRM structure...")
    if hasattr(model, 'lrm'):
        print(f"✅ LRM found: {type(model.lrm)}")
        print(f"   - LRM enabled: {getattr(model.lrm, 'enabled', 'Unknown')}")
        print(f"   - Disable modulation during inference: {getattr(model.lrm, 'disable_modulation_during_inference', 'Unknown')}")
        
        # Check mod_connections
        print(f"\n📋 Mod connections:")
        for i, conn in enumerate(model.lrm.mod_connections):
            print(f"   {i+1}. {conn['source']} -> {conn['destination']}")
        
        # Check LRM children (modulation layers)
        print(f"\n🏗️  LRM children (modulation layers):")
        for i, (name, child) in enumerate(model.lrm.named_children()):
            print(f"   {i+1}. {name}: {type(child)}")
            if hasattr(child, 'mod_inputs'):
                print(f"      - mod_inputs keys: {list(child.mod_inputs.keys())}")
            if hasattr(child, 'active_connections'):
                print(f"      - active_connections: {child.active_connections}")
        
        # Check if hooks are registered
        print(f"\n🎣 Hook status:")
        print(f"   - Target hooks: {len(model.lrm.targ_hooks)}")
        print(f"   - Mod hooks: {len(model.lrm.mod_hooks)}")
        
    else:
        print("❌ LRM not found in model!")
        return
    
    # Load test sample
    print(f"\n📊 Loading test sample...")
    dataset = EmoSoundscapesDataset()
    validate_sampler = EmotionValidateSampler(hdf5_path=dataset_path, batch_size=1, train_ratio=0.7)
    validate_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_sampler=validate_sampler, 
        collate_fn=emotion_collate_fn, 
        num_workers=0, 
        pin_memory=False
    )
    
    # Get test sample
    sample_data = next(iter(validate_loader))
    print(f"✅ Loaded test sample")
    
    # Prepare input
    feature = sample_data['feature']
    if len(feature.shape) == 3:
        feature = feature.unsqueeze(1)
    feature = feature.to(device)
    
    # Test single pass first
    print(f"\n🧪 Testing single pass...")
    with torch.no_grad():
        # Clear any stored activations
        if hasattr(model, 'lrm'):
            model.lrm.clear_stored_activations()
        
        output_1 = model(feature, forward_passes=1)
        valence_1 = output_1['valence'].cpu().numpy()[0, 0]
        arousal_1 = output_1['arousal'].cpu().numpy()[0, 0]
    
    print(f"   Single pass output: Valence={valence_1:.4f}, Arousal={arousal_1:.4f}")
    
    # Check what's in mod_inputs after single pass
    print(f"\n📦 Checking mod_inputs after single pass...")
    for name, child in model.lrm.named_children():
        if hasattr(child, 'mod_inputs'):
            print(f"   {name}: {len(child.mod_inputs)} inputs")
            for key, value in child.mod_inputs.items():
                print(f"     - {key}: {value.shape}")
    
    # Now test multiple passes and debug step by step
    print(f"\n🧪 Testing multiple passes with detailed debugging...")
    
    # Clear activations
    if hasattr(model, 'lrm'):
        model.lrm.clear_stored_activations()
    
    # Step 1: First pass
    print(f"\n   Step 1: First pass")
    with torch.no_grad():
        # Extract 128D representations manually to see what we're working with
        # Handle input dimensions properly
        if feature.dim() == 5:
            # DataParallel case: (num_gpus, batch_size, time_steps, mel_bins)
            original_shape = feature.shape
            feature_4d = feature.view(-1, original_shape[-2], original_shape[-1])
        elif feature.dim() == 4 and feature.shape[1] == 1:
            # Extra dimension case: (batch_size, 1, time_steps, mel_bins)
            feature_4d = feature.squeeze(1)  # (batch_size, time_steps, mel_bins)
        else:
            feature_4d = feature
            
        visual_embedding = model._forward_visual_system(feature_4d, mixup_lambda=None)
        
        valence_256d = model.affective_valence[0:2](visual_embedding)
        valence_128d = model.affective_valence[2:4](valence_256d)
        valence_out = model.affective_valence[4](valence_128d)
        
        arousal_256d = model.affective_arousal[0:2](visual_embedding)
        arousal_128d = model.affective_arousal[2:4](arousal_256d)
        arousal_out = model.affective_arousal[4](arousal_128d)
        
        print(f"     Visual embedding shape: {visual_embedding.shape}")
        print(f"     Valence 128D shape: {valence_128d.shape}")
        print(f"     Arousal 128D shape: {arousal_128d.shape}")
        print(f"     Valence output: {valence_out.item():.4f}")
        print(f"     Arousal output: {arousal_out.item():.4f}")
    
    # Step 2: Store feedback signals
    print(f"\n   Step 2: Storing feedback signals")
    model._store_feedback_signals(valence_128d, arousal_128d)
    
    # Check what was stored
    print(f"     Checking stored feedback signals:")
    for name, child in model.lrm.named_children():
        if hasattr(child, 'mod_inputs'):
            print(f"       {name}: {len(child.mod_inputs)} inputs")
            for key, value in child.mod_inputs.items():
                print(f"         - {key}: {value.shape}")
    
    # Step 3: Second pass
    print(f"\n   Step 3: Second pass")
    with torch.no_grad():
        output_2 = model(feature, forward_passes=2)
        valence_2 = output_2['valence'].cpu().numpy()[0, 0]
        arousal_2 = output_2['arousal'].cpu().numpy()[0, 0]
    
    print(f"     Second pass output: Valence={valence_2:.4f}, Arousal={arousal_2:.4f}")
    
    # Check if predictions changed
    valence_diff = abs(valence_2 - valence_1)
    arousal_diff = abs(arousal_2 - arousal_1)
    
    print(f"     Changes: Valence Δ={valence_diff:.6f}, Arousal Δ={arousal_diff:.6f}")
    
    # Check if modulation was actually applied
    print(f"\n🔍 Checking if modulation was applied...")
    for name, child in model.lrm.named_children():
        if hasattr(child, 'pre_mod_output') and hasattr(child, 'post_mod_output'):
            pre_mod = child.pre_mod_output
            post_mod = child.post_mod_output
            if pre_mod is not None and post_mod is not None:
                mod_diff = torch.abs(post_mod - pre_mod).max().item()
                print(f"   {name}: Max modulation difference = {mod_diff:.6f}")
                if mod_diff > 1e-6:
                    print(f"     ✅ Modulation was applied!")
                else:
                    print(f"     ❌ No modulation detected")
    
    # Check if disable_modulation_during_inference is True
    print(f"\n🔍 Checking modulation disable flags...")
    print(f"   LRM disable_modulation_during_inference: {getattr(model.lrm, 'disable_modulation_during_inference', 'Unknown')}")
    for name, child in model.lrm.named_children():
        if hasattr(child, 'disable_modulation_during_inference'):
            print(f"   {name} disable_modulation_during_inference: {child.disable_modulation_during_inference}")
    
    # Try to enable feedback explicitly
    print(f"\n🔧 Trying to enable feedback explicitly...")
    model.enable_feedback()
    print(f"   After enable_feedback():")
    print(f"   LRM disable_modulation_during_inference: {getattr(model.lrm, 'disable_modulation_during_inference', 'Unknown')}")
    for name, child in model.lrm.named_children():
        if hasattr(child, 'disable_modulation_during_inference'):
            print(f"   {name} disable_modulation_during_inference: {child.disable_modulation_during_inference}")
    
    # Test again with explicit enable
    print(f"\n🧪 Testing with explicit feedback enable...")
    with torch.no_grad():
        # Clear activations
        model.lrm.clear_stored_activations()
        
        # First pass
        output_1_enabled = model(feature, forward_passes=1)
        valence_1_enabled = output_1_enabled['valence'].cpu().numpy()[0, 0]
        arousal_1_enabled = output_1_enabled['arousal'].cpu().numpy()[0, 0]
        
        # Second pass
        output_2_enabled = model(feature, forward_passes=2)
        valence_2_enabled = output_2_enabled['valence'].cpu().numpy()[0, 0]
        arousal_2_enabled = output_2_enabled['arousal'].cpu().numpy()[0, 0]
    
    print(f"   Enabled feedback results:")
    print(f"     Pass 1: Valence={valence_1_enabled:.4f}, Arousal={arousal_1_enabled:.4f}")
    print(f"     Pass 2: Valence={valence_2_enabled:.4f}, Arousal={arousal_2_enabled:.4f}")
    
    valence_diff_enabled = abs(valence_2_enabled - valence_1_enabled)
    arousal_diff_enabled = abs(arousal_2_enabled - arousal_1_enabled)
    
    print(f"     Changes: Valence Δ={valence_diff_enabled:.6f}, Arousal Δ={arousal_diff_enabled:.6f}")
    
    print(f"\n🎯 Summary:")
    if valence_diff_enabled < 1e-6 and arousal_diff_enabled < 1e-6:
        print(f"   ❌ Feedback mechanism is still not working even after explicit enable")
        print(f"   🔍 Possible issues:")
        print(f"      - LRM hooks not properly registered")
        print(f"      - Feedback signals not reaching target layers")
        print(f"      - Modulation strength too low")
        print(f"      - Target layers not being modulated")
    else:
        print(f"   ✅ Feedback mechanism is working after explicit enable!")

def main():
    parser = argparse.ArgumentParser(description='Debug LRM connections and feedback signals')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to emotion features HDF5 file')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of samples to test')
    
    args = parser.parse_args()
    
    debug_lrm_connections(args.model_path, args.dataset_path, args.num_samples)

if __name__ == '__main__':
    main() 