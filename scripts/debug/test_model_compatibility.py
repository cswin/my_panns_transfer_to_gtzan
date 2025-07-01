#!/usr/bin/env python3
"""
Test script to compare model compatibility between training and steering contexts
"""

import os
import sys
import torch
import inspect

# Add src to path (like our steering scripts)
sys.path.append('src')

def test_model_compatibility():
    """Compare model imports and signatures between training and steering contexts."""
    
    print("🔍 Model Compatibility Test")
    print("=" * 50)
    
    # Test 1: Import from both paths
    print("\n📦 Test 1: Import Comparison")
    
    # Import like training script
    try:
        from src.models.emotion_models import FeatureEmotionRegression_Cnn6_LRM as TrainingModel
        print(f"✅ Training import: {TrainingModel.__module__}")
    except ImportError as e:
        print(f"❌ Training import failed: {e}")
        return
    
    # Import like steering scripts  
    try:
        from models.emotion_models import FeatureEmotionRegression_Cnn6_LRM as SteeringModel
        print(f"✅ Steering import: {SteeringModel.__module__}")
    except ImportError as e:
        print(f"❌ Steering import failed: {e}")
        return
    
    # Test 2: Check if they're the same class
    print(f"\n🔍 Test 2: Class Identity")
    print(f"   Same class: {TrainingModel is SteeringModel}")
    print(f"   Training class: {TrainingModel}")
    print(f"   Steering class: {SteeringModel}")
    
    # Test 3: Compare forward method signatures
    print(f"\n🔍 Test 3: Forward Method Signatures")
    
    training_sig = inspect.signature(TrainingModel.forward)
    steering_sig = inspect.signature(SteeringModel.forward)
    
    print(f"   Training forward: {list(training_sig.parameters.keys())}")
    print(f"   Steering forward: {list(steering_sig.parameters.keys())}")
    print(f"   Signatures match: {training_sig == steering_sig}")
    
    # Test 4: Create models and compare their structure
    print(f"\n🔍 Test 4: Model Structure Comparison")
    
    # Use same parameters as training
    from src.utils.config import sample_rate, cnn6_config
    
    training_model = TrainingModel(
        sample_rate=sample_rate,
        window_size=cnn6_config['window_size'],
        hop_size=cnn6_config['hop_size'],
        mel_bins=cnn6_config['mel_bins'],
        fmin=cnn6_config['fmin'],
        fmax=cnn6_config['fmax'],
        freeze_base=True,
        forward_passes=2
    )
    
    steering_model = SteeringModel(
        sample_rate=sample_rate,
        window_size=cnn6_config['window_size'],
        hop_size=cnn6_config['hop_size'],
        mel_bins=cnn6_config['mel_bins'],
        fmin=cnn6_config['fmin'],
        fmax=cnn6_config['fmax'],
        freeze_base=True,
        forward_passes=2
    )
    
    # Compare key attributes
    print(f"   Training model has LRM: {hasattr(training_model, 'lrm')}")
    print(f"   Steering model has LRM: {hasattr(steering_model, 'lrm')}")
    
    if hasattr(training_model, 'lrm') and hasattr(steering_model, 'lrm'):
        training_lrm_modules = list(training_model.lrm.named_children())
        steering_lrm_modules = list(steering_model.lrm.named_children())
        
        print(f"   Training LRM modules: {len(training_lrm_modules)}")
        print(f"   Steering LRM modules: {len(steering_lrm_modules)}")
        
        print(f"   Training LRM module names: {[name for name, _ in training_lrm_modules]}")
        print(f"   Steering LRM module names: {[name for name, _ in steering_lrm_modules]}")
    
    # Test 5: Check method availability
    print(f"\n🔍 Test 5: Method Availability")
    
    methods_to_check = ['add_steering_signal', '_apply_steering_signals', 'adjust_modulation_strengths']
    
    for method in methods_to_check:
        training_has = hasattr(training_model, method)
        steering_has = hasattr(steering_model, method)
        print(f"   {method}:")
        print(f"     Training: {training_has}")
        print(f"     Steering: {steering_has}")
        print(f"     Match: {training_has == steering_has}")
    
    # Test 6: Load checkpoint and test compatibility
    print(f"\n🔍 Test 6: Checkpoint Compatibility")
    
    checkpoint_paths = [
        '/home/pengliu/Private/my_panns_transfer_to_gtzan/workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth',
        'workspaces/emotion_feedback/checkpoints/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/best_model.pth'
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path:
        print(f"   Found checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Try loading into training model
            training_model.load_state_dict(checkpoint['model'])
            print("   ✅ Training model can load checkpoint")
            
            # Try loading into steering model  
            steering_model.load_state_dict(checkpoint['model'])
            print("   ✅ Steering model can load checkpoint")
            
        except Exception as e:
            print(f"   ❌ Checkpoint loading failed: {e}")
    else:
        print("   ❌ No checkpoint found")
    
    print(f"\n📊 Summary:")
    print(f"   Both imports work: ✅")
    print(f"   Same class: {'✅' if TrainingModel is SteeringModel else '❌'}")
    print(f"   Same signatures: {'✅' if training_sig == steering_sig else '❌'}")
    print(f"   Both have LRM: {'✅' if hasattr(training_model, 'lrm') and hasattr(steering_model, 'lrm') else '❌'}")

if __name__ == "__main__":
    test_model_compatibility() 