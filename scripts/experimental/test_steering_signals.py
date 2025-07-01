#!/usr/bin/env python3
"""
Test Steering Signals with Emotion Model

This script loads the generated steering signals and tests them with the emotion model
to demonstrate how external feedback can influence emotion predictions.

Usage:
    python scripts/test_steering_signals.py --model_checkpoint path/to/model.pth --steering_dir steering_signals --dataset_path features/emotion_features.h5
"""

import os
import argparse
import numpy as np
import torch
import h5py
import json
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn as nn

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append('src')

from data.data_generator import EmoSoundscapesDataset, EmotionValidateSampler, emotion_collate_fn
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
from utils.config import cnn6_config


def load_steering_signals(steering_dir):
    """Load steering signals from directory."""
    # Load category information
    category_info_path = os.path.join(steering_dir, "category_info.json")
    with open(category_info_path, 'r') as f:
        category_info = json.load(f)
    
    # Load steering signals
    steering_signals = {}
    for category_name in category_info['descriptions'].values():
        cat_dir = os.path.join(steering_dir, category_name)
        if os.path.exists(cat_dir):
            steering_signals[category_name] = {}
            for signal_file in os.listdir(cat_dir):
                if signal_file.endswith('.npy'):
                    signal_type = signal_file[:-4]  # Remove .npy
                    signal_path = os.path.join(cat_dir, signal_file)
                    signal_data = np.load(signal_path)
                    steering_signals[category_name][signal_type] = signal_data
    
    return steering_signals, category_info


def load_emotion_model(checkpoint_path, device):
    """Load emotion model from checkpoint."""
    # Load checkpoint with weights_only=False to handle older checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine model type from checkpoint or path
    model_type = None
    if 'model_type' in checkpoint:
        model_type = checkpoint['model_type']
    else:
        # Try to infer from path
        if 'LRM' in checkpoint_path:
            model_type = 'FeatureEmotionRegression_Cnn6_LRM'
        elif 'NewAffective' in checkpoint_path:
            model_type = 'FeatureEmotionRegression_Cnn6_NewAffective'
        else:
            model_type = 'FeatureEmotionRegression_Cnn6_LRM'  # Default
    
    # Create model based on type
    if model_type == 'FeatureEmotionRegression_Cnn6_LRM':
        # Use imported cnn6_config
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
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load model weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)
    
    # Move to device
    model = model.to(device)
    
    print(f"Loaded {model_type} model from {checkpoint_path}")
    return model


def select_steering_signal_by_target(target_valence, target_arousal, steering_signals):
    """
    Select appropriate steering signal based on target emotion labels.
    
    Args:
        target_valence: float, target valence value (-1 to 1)
        target_arousal: float, target arousal value (-1 to 1)
        steering_signals: dict, available steering signals
    
    Returns:
        tuple: (category_name, signals_dict) or (None, None) if not found
    """
    # Define thresholds for categorization (same as in generate_steering_signals.py)
    # Using quantile-based approach: negative < -0.33, neutral -0.33 to 0.33, positive > 0.33
    valence_thresholds = [-0.33, 0.33]
    arousal_thresholds = [-0.33, 0.33]
    
    # Categorize valence
    if target_valence < valence_thresholds[0]:
        valence_category = "negative"
    elif target_valence > valence_thresholds[1]:
        valence_category = "positive"
    else:
        valence_category = "neutral"
    
    # Categorize arousal
    if target_arousal < arousal_thresholds[0]:
        arousal_category = "weak"
    elif target_arousal > arousal_thresholds[1]:
        arousal_category = "strong"
    else:
        arousal_category = "middle"
    
    # Construct category name
    category_name = f"{valence_category}_{arousal_category}"
    
    # Check if this category exists in steering signals
    if category_name in steering_signals:
        return category_name, steering_signals[category_name]
    else:
        print(f"Warning: Steering signal category '{category_name}' not found")
        return None, None


def test_steering_on_sample(model, sample_data, steering_signals, device, steering_strength=1.0, steering_method='lrm'):
    """Test steering signals on a single sample with different methods."""
    model.eval()
    
    # Debug: Check model state
    print(f"    Model training mode: {model.training}")
    print(f"    LRM disabled: {model.lrm.disable_modulation_during_inference}")
    
    # Prepare input
    feature = sample_data['feature'].unsqueeze(0).to(device)  # Add batch dimension
    if len(feature.shape) == 3:
        feature = feature.unsqueeze(1)  # Add channel dimension
    
    results = {}
    
    # Baseline prediction (no steering)
    with torch.no_grad():
        baseline_output = model(feature, forward_passes=1)
        results['baseline'] = {
            'valence': baseline_output['valence'].cpu().numpy().flatten()[0],
            'arousal': baseline_output['arousal'].cpu().numpy().flatten()[0]
        }
    
    # Get target emotion labels
    target_valence = sample_data['valence'].item()
    target_arousal = sample_data['arousal'].item()
    
    print(f"    Target emotion - Valence: {target_valence:.3f}, Arousal: {target_arousal:.3f}")
    
    # Select appropriate steering signal based on target
    category_name, signals = select_steering_signal_by_target(target_valence, target_arousal, steering_signals)
    
    if category_name is None or signals is None:
        print(f"    No appropriate steering signal found for target emotion")
        return results
    
    print(f"    Selected steering category: {category_name}")
    print(f"    Available signals: {list(signals.keys())}")
    
    if 'valence_128d' in signals and 'arousal_128d' in signals:
        # Convert numpy arrays to tensors (keep raw signals)
        valence_128d = torch.from_numpy(signals['valence_128d']).float().to(device)
        arousal_128d = torch.from_numpy(signals['arousal_128d']).float().to(device)
        
        # Note: amplification will be handled by add_steering_signal() internally
        
        # Debug: Print steering signal statistics
        print(f"    Applying {category_name} steering signals:")
        print(f"      Valence 128D - mean: {valence_128d.mean():.4f}, std: {valence_128d.std():.4f}, range: [{valence_128d.min():.4f}, {valence_128d.max():.4f}]")
        print(f"      Arousal 128D - mean: {arousal_128d.mean():.4f}, std: {arousal_128d.std():.4f}, range: [{arousal_128d.min():.4f}, {arousal_128d.max():.4f}]")
        
        # Apply steering signals using different methods
        if steering_method == 'lrm':
            # Method 1: Standard LRM approach
            model.set_external_feedback(valence_128d=valence_128d, arousal_128d=arousal_128d)
            
        elif steering_method == 'direct':
            # Method 2: Direct injection into model's internal feedback
            model.valence_128d = valence_128d.unsqueeze(-1).unsqueeze(-1)  # (batch, 128, 1, 1)
            model.arousal_128d = arousal_128d.unsqueeze(-1).unsqueeze(-1)  # (batch, 128, 1, 1)
            
        elif steering_method == 'bypass':
            # Method 3: Bypass normalization and squashing
            model.set_external_feedback(valence_128d=valence_128d, arousal_128d=arousal_128d)
            # Temporarily bypass normalization and squashing in ModBlocks
            for mod_name, mod_block in model.lrm.named_children():
                if hasattr(mod_block, 'rescale'):
                    mod_block._original_rescale = mod_block.rescale
                    # Create an identity module that just returns the input
                    class IdentityRescale(nn.Module):
                        def forward(self, x, out_size):
                            return x
                    mod_block.rescale = IdentityRescale()
        
        elif steering_method == 'modulation':
            # Method 4: Use modulation strength control
            model.set_external_feedback(valence_128d=valence_128d, arousal_128d=arousal_128d)
            model.set_modulation_strength(10.0)  # High modulation strength
        
        elif steering_method == 'external':
            # Method 5: Use external steering format (like the tmp/ example code)
            # Let add_steering_signal handle amplification internally (like tmp/ code)
            model.add_steering_signal(
                source='affective_valence_128d',
                activation=valence_128d,  # Raw signal
                strength=steering_strength,  # Amplification happens here
                alpha=1.0
            )
            model.add_steering_signal(
                source='affective_arousal_128d', 
                activation=arousal_128d,  # Raw signal
                strength=steering_strength,  # Amplification happens here
                alpha=1.0
            )
        
        # Debug: Check if external feedback was stored
        print(f"      LRM mod_inputs keys: {list(model.lrm.mod_inputs.keys())}")
        if model.lrm.mod_inputs:
            for key, value in model.lrm.mod_inputs.items():
                print(f"        {key}: shape {value.shape}, mean {value.mean():.4f}")
        
        # Debug: Check individual LRM modules
        print(f"      Individual LRM module mod_inputs:")
        for lrm_module_name, lrm_module in model.lrm.named_children():
            if hasattr(lrm_module, 'mod_inputs') and lrm_module.mod_inputs:
                print(f"        {lrm_module_name}: {list(lrm_module.mod_inputs.keys())}")
                for key, value in lrm_module.mod_inputs.items():
                    print(f"          {key}: shape {value.shape}, mean {value.mean():.4f}")
            else:
                print(f"        {lrm_module_name}: empty")
        
        # Check if LRM is enabled
        print(f"      LRM disabled: {model.lrm.disable_modulation_during_inference}")
        
        # Enable LRM explicitly
        model.lrm.enable()
        print(f"      LRM enabled: {not model.lrm.disable_modulation_during_inference}")
        
        if steering_method == 'modulation':
            print(f"      Modulation strength increased to 10.0")
        
        # Get prediction with steering
        with torch.no_grad():
            if steering_method == 'external':
                # For external method, we already called add_steering_signal directly
                steered_output = model(feature, forward_passes=2)
            else:
                steered_output = model(feature, forward_passes=2)
            
            results[category_name] = {
                'valence': steered_output['valence'].cpu().numpy().flatten()[0],
                'arousal': steered_output['arousal'].cpu().numpy().flatten()[0]
            }
        
        # Cleanup based on steering method
        if steering_method == 'bypass':
            # Restore original rescale functions
            for mod_name, mod_block in model.lrm.named_children():
                if hasattr(mod_block, 'rescale') and hasattr(mod_block, '_original_rescale'):
                    mod_block.rescale = mod_block._original_rescale
        elif steering_method == 'modulation':
            model.reset_modulation_strength()
        
        # Clear feedback state
        model.clear_feedback_state()
    else:
        print(f"    Skipping {category_name} - missing required signals")
    
    return results


def test_all_steering_methods(model, sample_data, steering_signals, device, steering_strength=1.0):
    """Test all steering methods on a single sample and compare results."""
    model.eval()
    
    print(f"\n=== COMPREHENSIVE STEERING SIGNAL TEST ===")
    print(f"Testing all steering methods with strength {steering_strength}x")
    
    # Prepare input
    feature = sample_data['feature'].unsqueeze(0).to(device)
    if len(feature.shape) == 3:
        feature = feature.unsqueeze(1)
    
    # Baseline prediction
    with torch.no_grad():
        baseline_output = model(feature, forward_passes=1)
        baseline_valence = baseline_output['valence'].cpu().numpy().flatten()[0]
        baseline_arousal = baseline_output['arousal'].cpu().numpy().flatten()[0]
    
    print(f"\nBaseline prediction:")
    print(f"  Valence: {baseline_valence:.4f}")
    print(f"  Arousal: {baseline_arousal:.4f}")
    
    # Test each steering method
    methods = ['lrm', 'direct', 'bypass', 'modulation', 'external']
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} method ---")
        
        # Get steering signal for positive_strong category
        if 'positive_strong' in steering_signals:
            signals = steering_signals['positive_strong']
            valence_128d = torch.from_numpy(signals['valence_128d']).float().to(device) * steering_strength
            arousal_128d = torch.from_numpy(signals['arousal_128d']).float().to(device) * steering_strength
            
            print(f"  Steering signal stats:")
            print(f"    Valence: mean={valence_128d.mean():.4f}, std={valence_128d.std():.4f}")
            print(f"    Arousal: mean={arousal_128d.mean():.4f}, std={arousal_128d.std():.4f}")
            
            # Apply steering using the method
            if method == 'lrm':
                model.set_external_feedback(valence_128d=valence_128d, arousal_128d=arousal_128d)
            elif method == 'direct':
                model.valence_128d = valence_128d.unsqueeze(-1).unsqueeze(-1)
                model.arousal_128d = arousal_128d.unsqueeze(-1).unsqueeze(-1)
            elif method == 'bypass':
                model.set_external_feedback(valence_128d=valence_128d, arousal_128d=arousal_128d)
                for mod_name, mod_block in model.lrm.named_children():
                    if hasattr(mod_block, 'rescale'):
                        mod_block._original_rescale = mod_block.rescale
                        class IdentityRescale(nn.Module):
                            def forward(self, x, out_size):
                                return x
                        mod_block.rescale = IdentityRescale()
            elif method == 'modulation':
                model.set_external_feedback(valence_128d=valence_128d, arousal_128d=arousal_128d)
                model.set_modulation_strength(10.0)
            
            elif method == 'external':
                # Method 5: Use external steering format (like the example code)
                # Call add_steering_signal directly
                model.add_steering_signal(
                    source='affective_valence_128d',
                    activation=valence_128d,
                    strength=steering_strength,
                    alpha=1.0
                )
                model.add_steering_signal(
                    source='affective_arousal_128d', 
                    activation=arousal_128d,
                    strength=steering_strength,
                    alpha=1.0
                )
            
            # Enable LRM
            model.lrm.enable()
            
            # Get prediction
            with torch.no_grad():
                if method == 'external':
                    # For external method, we already called add_steering_signal directly
                    steered_output = model(feature, forward_passes=2)
                else:
                    steered_output = model(feature, forward_passes=2)
                
                steered_valence = steered_output['valence'].cpu().numpy().flatten()[0]
                steered_arousal = steered_output['arousal'].cpu().numpy().flatten()[0]
            
            # Calculate changes
            valence_change = steered_valence - baseline_valence
            arousal_change = steered_arousal - baseline_arousal
            
            print(f"  Results:")
            print(f"    Valence: {steered_valence:.4f} (change: {valence_change:+.4f})")
            print(f"    Arousal: {steered_arousal:.4f} (change: {arousal_change:+.4f})")
            
            results[method] = {
                'valence': steered_valence,
                'arousal': steered_arousal,
                'valence_change': valence_change,
                'arousal_change': arousal_change
            }
            
            # Cleanup
            if method == 'bypass':
                for mod_name, mod_block in model.lrm.named_children():
                    if hasattr(mod_block, 'rescale') and hasattr(mod_block, '_original_rescale'):
                        mod_block.rescale = mod_block._original_rescale
            elif method == 'modulation':
                model.reset_modulation_strength()
            
            model.clear_feedback_state()
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Method          | Valence Change | Arousal Change")
    print(f"----------------|----------------|---------------")
    for method in methods:
        if method in results:
            v_change = results[method]['valence_change']
            a_change = results[method]['arousal_change']
            print(f"{method:14} | {v_change:+8.4f}     | {a_change:+8.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Test steering signals with emotion model')
    parser.add_argument('--model_checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--steering_dir', type=str, required=True,
                       help='Directory containing steering signals')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to emotion_features.h5 dataset file')
    parser.add_argument('--output_dir', type=str, default='steering_test_results',
                       help='Output directory for test results')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to test')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for processing')
    parser.add_argument('--cuda', action='store_true',
                       help='Use CUDA if available')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--steering_strength', type=float, default=1.0,
                       help='Amplification factor for steering signals (default: 1.0)')
    parser.add_argument('--steering_method', type=str, default='external',
                       choices=['lrm', 'direct', 'bypass', 'modulation', 'external'],
                       help='Steering method to use (default: external)')
    parser.add_argument('--comprehensive_test', action='store_true',
                       help='Run comprehensive test comparing all steering methods')
    
    args = parser.parse_args()
    
    # Set device
    if args.cuda and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
        print(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load steering signals
    print(f"Loading steering signals from {args.steering_dir}")
    steering_signals, category_info = load_steering_signals(args.steering_dir)
    print(f"Loaded {len(steering_signals)} steering signal categories")
    
    # Print steering signal statistics
    print(f"\nSteering signal statistics (before amplification):")
    for category_name, signals in steering_signals.items():
        if 'valence_128d' in signals and 'arousal_128d' in signals:
            valence_stats = signals['valence_128d']
            arousal_stats = signals['arousal_128d']
            print(f"  {category_name}:")
            print(f"    Valence - mean: {np.mean(valence_stats):.4f}, std: {np.std(valence_stats):.4f}, range: [{np.min(valence_stats):.4f}, {np.max(valence_stats):.4f}]")
            print(f"    Arousal - mean: {np.mean(arousal_stats):.4f}, std: {np.std(arousal_stats):.4f}, range: [{np.min(arousal_stats):.4f}, {np.max(arousal_stats):.4f}]")
    
    print(f"\nSteering strength amplification factor: {args.steering_strength}")
    print(f"After amplification, signals will be {args.steering_strength}x stronger")
    
    # Load model
    print(f"Loading model from {args.model_checkpoint}")
    model = load_emotion_model(args.model_checkpoint, device)
    
    # Create data loader (use same validation split as training: 70% train, 30% val)
    print("Creating data loader...")
    dataset = EmoSoundscapesDataset()
    sampler = EmotionValidateSampler(args.dataset_path, args.batch_size, train_ratio=0.7)
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        collate_fn=emotion_collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Test steering signals on a few samples
    print(f"Testing steering signals on {args.num_samples} samples...")
    all_results = []
    
    if args.comprehensive_test:
        # Run comprehensive test on first sample only
        print("Running comprehensive steering method comparison...")
        for i, batch_data in enumerate(data_loader):
            if i >= 1:  # Only test first sample
                break
                
            sample_data = {
                'feature': batch_data['feature'][0],
                'valence': batch_data['valence'][0],
                'arousal': batch_data['arousal'][0],
                'audio_name': batch_data['audio_name'][0]
            }
            
            print(f"Testing sample: {sample_data['audio_name']}")
            
            # Run comprehensive test
            comprehensive_results = test_all_steering_methods(
                model, sample_data, steering_signals, device, args.steering_strength
            )
            
            # Save comprehensive results
            all_results.append({
                'audio_name': sample_data['audio_name'],
                'target': {
                    'valence': sample_data['valence'].item(),
                    'arousal': sample_data['arousal'].item()
                },
                'comprehensive_results': comprehensive_results
            })
    else:
        # Standard testing with automatic steering signal selection
        print("\n" + "="*60)
        print("TESTING AUTOMATIC STEERING SIGNAL SELECTION")
        print("="*60)
        print("Each sample will automatically select the most appropriate")
        print("steering signal based on its target emotion labels.")
        print("="*60)
        
        sample_count = 0
        for i, batch_data in enumerate(data_loader):
            if args.num_samples > 0 and sample_count >= args.num_samples:
                break
                
            # Test each sample in the batch
            for j in range(len(batch_data['feature'])):
                if args.num_samples > 0 and sample_count >= args.num_samples:
                    break
                    
                sample_data = {
                    'feature': batch_data['feature'][j],
                    'valence': batch_data['valence'][j],
                    'arousal': batch_data['arousal'][j],
                    'audio_name': batch_data['audio_name'][j]
                }
                
                sample_count += 1
                print(f"\nSample {sample_count}/{args.num_samples}: {sample_data['audio_name']}")
                print(f"Target emotion - Valence: {sample_data['valence'].item():.3f}, Arousal: {sample_data['arousal'].item():.3f}")
                
                # Show which steering signal category would be selected
                target_valence = sample_data['valence'].item()
                target_arousal = sample_data['arousal'].item()
                category_name, _ = select_steering_signal_by_target(target_valence, target_arousal, steering_signals)
                
                if category_name:
                    print(f"Auto-selected steering category: {category_name}")
                else:
                    print(f"No appropriate steering signal found")
                
                # Test steering signals using automatic selection
                results = test_steering_on_sample(model, sample_data, steering_signals, device, args.steering_strength, args.steering_method)
                
                # Add target values
                results['target'] = {
                    'valence': sample_data['valence'].item(),
                    'arousal': sample_data['arousal'].item()
                }
                results['audio_name'] = sample_data['audio_name']
                results['selected_category'] = category_name
                
                # Show effectiveness of steering
                if 'baseline' in results and category_name and category_name in results:
                    baseline = results['baseline']
                    steered = results[category_name]
                    
                    val_change = steered['valence'] - baseline['valence']
                    aro_change = steered['arousal'] - baseline['arousal']
                    
                    # Calculate if steering moved in the correct direction
                    val_direction_correct = (val_change > 0 and target_valence > baseline['valence']) or (val_change < 0 and target_valence < baseline['valence']) or abs(val_change) < 0.001
                    aro_direction_correct = (aro_change > 0 and target_arousal > baseline['arousal']) or (aro_change < 0 and target_arousal < baseline['arousal']) or abs(aro_change) < 0.001
                    
                    direction_indicator = ""
                    if val_direction_correct and aro_direction_correct:
                        direction_indicator = " ✓ (both directions correct)"
                    elif val_direction_correct or aro_direction_correct:
                        direction_indicator = " ~ (partially correct)"
                    else:
                        direction_indicator = " ✗ (incorrect directions)"
                    
                    print(f"Results:")
                    print(f"  Baseline:  Valence {baseline['valence']:+.4f}, Arousal {baseline['arousal']:+.4f}")
                    print(f"  Steered:   Valence {steered['valence']:+.4f}, Arousal {steered['arousal']:+.4f}")
                    print(f"  Changes:   Valence {val_change:+.4f}, Arousal {aro_change:+.4f}{direction_indicator}")
                
                all_results.append(results)
    
    # Save results
    print(f"Saving results to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save detailed results as JSON
    results_path = os.path.join(args.output_dir, "steering_test_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = []
    for result in all_results:
        json_result = {}
        for key, value in result.items():
            if key == 'audio_name':
                json_result[key] = value
            elif key == 'comprehensive_results':
                # Handle comprehensive results specially
                json_result[key] = {}
                for method, method_results in value.items():
                    json_result[key][method] = {k: float(v) for k, v in method_results.items()}
            elif isinstance(value, dict):
                json_result[key] = {k: float(v) for k, v in value.items()}
        json_results.append(json_result)
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    # Print summary
    print(f"\nSteering signal testing completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Tested {len(all_results)} samples")
    
    # Analyze automatic steering effectiveness
    if all_results and not args.comprehensive_test:
        print("\n" + "="*60)
        print("AUTOMATIC STEERING SELECTION EFFECTIVENESS SUMMARY")
        print("="*60)
        
        successful_selections = 0
        total_samples = 0
        valence_correct_direction = 0
        arousal_correct_direction = 0
        both_correct_direction = 0
        category_usage = {}
        
        for result in all_results:
            if 'selected_category' in result and result['selected_category'] and 'baseline' in result:
                total_samples += 1
                category = result['selected_category']
                
                # Count category usage
                category_usage[category] = category_usage.get(category, 0) + 1
                
                if category in result:
                    successful_selections += 1
                    
                    # Check direction correctness
                    baseline = result['baseline']
                    steered = result[category]
                    target = result['target']
                    
                    val_change = steered['valence'] - baseline['valence']
                    aro_change = steered['arousal'] - baseline['arousal']
                    
                    val_direction_correct = (val_change > 0 and target['valence'] > baseline['valence']) or (val_change < 0 and target['valence'] < baseline['valence']) or abs(val_change) < 0.001
                    aro_direction_correct = (aro_change > 0 and target['arousal'] > baseline['arousal']) or (aro_change < 0 and target['arousal'] < baseline['arousal']) or abs(aro_change) < 0.001
                    
                    if val_direction_correct:
                        valence_correct_direction += 1
                    if aro_direction_correct:
                        arousal_correct_direction += 1
                    if val_direction_correct and aro_direction_correct:
                        both_correct_direction += 1
        
        if total_samples > 0:
            print(f"Selection Success Rate: {successful_selections}/{total_samples} ({100*successful_selections/total_samples:.1f}%)")
            print(f"Valence Direction Accuracy: {valence_correct_direction}/{successful_selections} ({100*valence_correct_direction/successful_selections:.1f}%)")
            print(f"Arousal Direction Accuracy: {arousal_correct_direction}/{successful_selections} ({100*arousal_correct_direction/successful_selections:.1f}%)")
            print(f"Both Directions Correct: {both_correct_direction}/{successful_selections} ({100*both_correct_direction/successful_selections:.1f}%)")
            
            print(f"\nSteering Category Usage:")
            for category, count in sorted(category_usage.items()):
                print(f"  {category}: {count} samples ({100*count/total_samples:.1f}%)")
            
            print(f"\nLegend:")
            print(f"  ✓ = steering moved both valence and arousal in correct directions")
            print(f"  ~ = steering moved at least one dimension in correct direction")
            print(f"  ✗ = steering moved both dimensions in wrong directions")
    
    # Print some example results
    if all_results:
        print(f"\n" + "="*60)
        print("EXAMPLE RESULTS")
        print("="*60)
        first_result = all_results[0]
        print(f"Audio: {first_result['audio_name']}")
        print(f"Target - Valence: {first_result['target']['valence']:.3f}, Arousal: {first_result['target']['arousal']:.3f}")
        print(f"Baseline - Valence: {first_result['baseline']['valence']:.3f}, Arousal: {first_result['baseline']['arousal']:.3f}")
        
        # Show steering effects for selected category
        if 'selected_category' in first_result and first_result['selected_category'] in first_result:
            cat = first_result['selected_category']
            valence_change = first_result[cat]['valence'] - first_result['baseline']['valence']
            arousal_change = first_result[cat]['arousal'] - first_result['baseline']['arousal']
            print(f"Selected category '{cat}' - Valence change: {valence_change:+.3f}, Arousal change: {arousal_change:+.3f}")
        else:
            print("No steering applied or category not found")


if __name__ == '__main__':
    main()
