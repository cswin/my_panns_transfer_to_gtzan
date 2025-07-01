#!/usr/bin/env python3

import sys
import os
# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import json
from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM

def debug_zero_steering():
    """Debug why steering effects are zero."""
    
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
    
    # Get first available category
    available_categories = list(signals_25bin.keys())
    category = None
    for cat in available_categories:
        if 'valence_128d' in signals_25bin[cat] and 'arousal_128d' in signals_25bin[cat]:
            category = cat
            break
    
    signals = signals_25bin[category]
    print(f"🎯 Using category: {category}")
    
    # Test sample
    sample_tensor = torch.randn(1, 1024, 64).to(device)
    
    print("\n" + "="*60)
    print("DEBUGGING ZERO STEERING EFFECTS")
    print("="*60)
    
    # Test 1: Baseline (no steering)
    print("\n🔍 Test 1: Baseline (no steering)")
    model.lrm.clear_stored_activations()
    model.lrm.disable()
    
    with torch.no_grad():
        baseline_output = model(sample_tensor, forward_passes=2)
    
    baseline_val = baseline_output['valence'].item()
    baseline_aro = baseline_output['arousal'].item()
    print(f"   Baseline: Valence={baseline_val:.6f}, Arousal={baseline_aro:.6f}")
    
    # Test 2: With steering (minimal debug output)
    print("\n🔍 Test 2: With steering (strength=5.0)")
    model.lrm.clear_stored_activations()
    
    # Apply steering WITHOUT debug output
    if 'valence_128d' in signals:
        valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
        # Temporarily disable debug output
        old_add_method = model.add_steering_signal
        
        def quiet_add_steering_signal(source, activation, strength, alpha=1):
            """Add steering signal without debug output."""
            neg_scale_adjust, pos_scale_adjust = torch.nn.modules.utils._pair(strength)
            
            pattern = f'from_{source.replace(".", "_")}_to_'
            found_matching_module = False
            
            for lrm_module_name, lrm_module in model.lrm.named_children():
                for feedback_module_name, feedback_module in lrm_module.named_children():
                    if pattern in feedback_module_name:
                        found_matching_module = True
                        # Adjust modulation strengths
                        model.adjust_modulation_strengths(feedback_module, neg_scale_adjust, pos_scale_adjust)

                        # Align the shape of the activation
                        if activation.dim() == 1:
                            activation = activation.unsqueeze(0)
                        elif activation.dim() == 2:
                            activation = activation.unsqueeze(-1).unsqueeze(-1)
                        elif activation.dim() == 3:
                            activation = activation.unsqueeze(0)

                        # Store the steering signal
                        lrm_module.mod_inputs[feedback_module_name] = activation
        
        # Use quiet method
        model.add_steering_signal = quiet_add_steering_signal
        model.add_steering_signal('affective_valence_128d', valence_signal, strength=5.0)
        
        # Restore original method
        model.add_steering_signal = old_add_method
    
    if 'arousal_128d' in signals:
        arousal_signal = torch.tensor(signals['arousal_128d'], dtype=torch.float32).to(device)
        model.add_steering_signal = quiet_add_steering_signal
        model.add_steering_signal('affective_arousal_128d', arousal_signal, strength=5.0)
        model.add_steering_signal = old_add_method
    
    # Check if signals were stored
    total_signals = 0
    for lrm_module_name, lrm_module in model.lrm.named_children():
        signals_in_module = len(lrm_module.mod_inputs)
        total_signals += signals_in_module
        print(f"   {lrm_module_name}: {signals_in_module} steering signals")
    
    print(f"   Total steering signals stored: {total_signals}")
    
    # Enable LRM
    model.lrm.enable()
    
    # Forward pass
    with torch.no_grad():
        steering_output = model(sample_tensor, forward_passes=2)
    
    steering_val = steering_output['valence'].item()
    steering_aro = steering_output['arousal'].item()
    
    val_change = steering_val - baseline_val
    aro_change = steering_aro - baseline_aro
    
    print(f"   With steering: Valence={steering_val:.6f}, Arousal={steering_aro:.6f}")
    print(f"   Changes: Valence Δ={val_change:+.6f}, Arousal Δ={aro_change:+.6f}")
    
    # Test 3: Check if LRM is actually enabled
    print("\n🔍 Test 3: LRM Status Check")
    print(f"   LRM disable_modulation_during_inference: {model.lrm.disable_modulation_during_inference}")
    for lrm_module_name, lrm_module in model.lrm.named_children():
        print(f"   {lrm_module_name} disable_modulation_during_inference: {lrm_module.disable_modulation_during_inference}")
    
    # Test 4: Check modulation statistics during forward pass
    print("\n🔍 Test 4: Hook Statistics")
    
    hook_stats = {}
    
    def debug_hook(module, input, output):
        target_name = None
        for name, mod in model.lrm.named_children():
            if mod is module:
                target_name = name
                break
        
        if target_name:
            hook_stats[target_name] = {
                'mod_inputs_count': len(module.mod_inputs),
                'has_total_mod': hasattr(module, 'total_mod') and module.total_mod is not None,
                'total_mod_stats': None
            }
            
            if hasattr(module, 'total_mod') and module.total_mod is not None:
                hook_stats[target_name]['total_mod_stats'] = {
                    'mean': module.total_mod.mean().item(),
                    'std': module.total_mod.std().item(),
                    'max': module.total_mod.max().item(),
                    'min': module.total_mod.min().item()
                }
    
    # Register hooks
    hooks = []
    for name, lrm_module in model.lrm.named_children():
        hooks.append(lrm_module.register_forward_hook(debug_hook))
    
    # Forward pass with hooks
    model.lrm.clear_stored_activations()
    
    # Re-apply steering
    if 'valence_128d' in signals:
        valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
        model.add_steering_signal = quiet_add_steering_signal
        model.add_steering_signal('affective_valence_128d', valence_signal, strength=5.0)
        model.add_steering_signal = old_add_method
    
    model.lrm.enable()
    
    with torch.no_grad():
        _ = model(sample_tensor, forward_passes=2)
    
    # Print hook statistics
    for target_name, stats in hook_stats.items():
        print(f"   {target_name}:")
        print(f"     mod_inputs_count: {stats['mod_inputs_count']}")
        print(f"     has_total_mod: {stats['has_total_mod']}")
        if stats['total_mod_stats']:
            mod_stats = stats['total_mod_stats']
            print(f"     total_mod: mean={mod_stats['mean']:.6f}, std={mod_stats['std']:.6f}, range=[{mod_stats['min']:.6f}, {mod_stats['max']:.6f}]")
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Test 5: Simple single-pass test
    print("\n🔍 Test 5: Single-pass test")
    model.lrm.clear_stored_activations()
    
    # Apply steering
    if 'valence_128d' in signals:
        valence_signal = torch.tensor(signals['valence_128d'], dtype=torch.float32).to(device)
        model.add_steering_signal = quiet_add_steering_signal
        model.add_steering_signal('affective_valence_128d', valence_signal, strength=10.0)
        model.add_steering_signal = old_add_method
    
    model.lrm.enable()
    
    with torch.no_grad():
        single_pass_output = model(sample_tensor, forward_passes=1)
    
    single_val = single_pass_output['valence'].item()
    single_aro = single_pass_output['arousal'].item()
    
    single_val_change = single_val - baseline_val
    single_aro_change = single_aro - baseline_aro
    
    print(f"   Single pass: Valence={single_val:.6f}, Arousal={single_aro:.6f}")
    print(f"   Changes: Valence Δ={single_val_change:+.6f}, Arousal Δ={single_aro_change:+.6f}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if abs(val_change) > 0.001 or abs(aro_change) > 0.001:
        print("✅ Steering is working!")
    elif abs(single_val_change) > 0.001 or abs(single_aro_change) > 0.001:
        print("⚠️ Steering works with single pass but not multi-pass")
    else:
        print("❌ Steering is not working")
        
        if total_signals == 0:
            print("   Issue: No steering signals stored")
        elif not any(stats['has_total_mod'] for stats in hook_stats.values()):
            print("   Issue: Forward hooks not creating total_mod")
        else:
            print("   Issue: Unknown - steering signals stored and hooks firing but no effect")

if __name__ == "__main__":
    debug_zero_steering() 