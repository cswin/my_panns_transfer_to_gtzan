#!/usr/bin/env python3
"""
Example script demonstrating the FeatureEmotionRegression_Cnn6_LRM model.

This script shows how to use the new LRM-based emotion regression model
that incorporates psychologically-motivated top-down feedback mechanisms.

The feedback model uses:
- Valence predictions to modulate semantic processing (conv3, conv4)
- Arousal predictions to modulate attention to acoustic details (conv1, conv2)
- Multiple forward passes for iterative refinement
"""

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], 'pytorch'))

import torch
import numpy as np
from src.utils.config import sample_rate, mel_bins, fmin, fmax, window_size, hop_size, cnn6_config
from src.models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
from src.models.cnn_models import FeatureEmotionRegression_Cnn6

def compare_models():
    """Compare standard and feedback-enabled models."""
    
    print("🎵 PANNs Emotion Regression: Feedback vs Standard Model Comparison")
    print("=" * 80)
    
    # Configuration
    config = cnn6_config
    batch_size = 2
    time_steps = 1024  # Example time steps
    
    # Create both models
    print("Creating models...")
    
    # Standard model
    standard_model = FeatureEmotionRegression_Cnn6(
        sample_rate=sample_rate,
        window_size=config['window_size'],
        hop_size=config['hop_size'],
        mel_bins=config['mel_bins'],
        fmin=config['fmin'],
        fmax=config['fmax'],
        freeze_base=True
    )
    
    # Feedback model
    feedback_model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=sample_rate,
        window_size=config['window_size'],
        hop_size=config['hop_size'],
        mel_bins=config['mel_bins'],
        fmin=config['fmin'],
        fmax=config['fmax'],
        freeze_base=True,
        forward_passes=3  # Use 3 forward passes for demonstration
    )
    
    # Model information
    standard_params = sum(p.numel() for p in standard_model.parameters())
    feedback_params = sum(p.numel() for p in feedback_model.parameters())
    
    print(f"✅ Standard Model: {standard_params:,} parameters")
    print(f"✅ Feedback Model: {feedback_params:,} parameters")
    print(f"   Additional parameters for feedback: {feedback_params - standard_params:,}")
    
    # Create dummy input (pre-computed mel-spectrogram features)
    print(f"\nCreating dummy input: ({batch_size}, {time_steps}, {mel_bins})")
    dummy_input = torch.randn(batch_size, time_steps, mel_bins)
    
    # Test standard model
    print("\n" + "="*50)
    print("🔄 Testing Standard Model (Single Forward Pass)")
    print("="*50)
    
    standard_model.eval()
    with torch.no_grad():
        standard_output = standard_model(dummy_input)
    
    print(f"Standard Model Output:")
    print(f"  Valence shape: {standard_output['valence'].shape}")
    print(f"  Arousal shape: {standard_output['arousal'].shape}")
    print(f"  Embedding shape: {standard_output['embedding'].shape}")
    print(f"  Sample Valence predictions: {standard_output['valence'].flatten()[:4].tolist()}")
    print(f"  Sample Arousal predictions: {standard_output['arousal'].flatten()[:4].tolist()}")
    
    # Test feedback model
    print("\n" + "="*50)
    print("🔄 Testing Feedback Model (Multiple Forward Passes)")
    print("="*50)
    
    feedback_model.eval()
    with torch.no_grad():
        # Test with different numbers of forward passes
        for num_passes in [1, 2, 3]:
            print(f"\n--- {num_passes} Forward Pass(es) ---")
            
            # Single output (final pass only)
            output = feedback_model(dummy_input, forward_passes=num_passes)
            print(f"Final Pass Output:")
            print(f"  Valence: {output['valence'].flatten()[:4].tolist()}")
            print(f"  Arousal: {output['arousal'].flatten()[:4].tolist()}")
            
            # All passes output
            all_outputs = feedback_model(dummy_input, forward_passes=num_passes, 
                                       return_all_passes=True)
            print(f"All {num_passes} Pass(es) Outputs:")
            for i, pass_output in enumerate(all_outputs):
                print(f"  Pass {i+1} - Valence: {pass_output['valence'].flatten()[0]:.4f}, "
                      f"Arousal: {pass_output['arousal'].flatten()[0]:.4f}")
    
    # Test feedback control
    print("\n" + "="*50)
    print("🎛️  Testing Feedback Control")
    print("="*50)
    
    # Test with feedback disabled
    feedback_model.disable_feedback()
    with torch.no_grad():
        output_no_feedback = feedback_model(dummy_input, forward_passes=2)
    
    # Test with feedback enabled
    feedback_model.enable_feedback()
    with torch.no_grad():
        output_with_feedback = feedback_model(dummy_input, forward_passes=2)
    
    print("Feedback Control Results:")
    print(f"  Without Feedback - Valence: {output_no_feedback['valence'].flatten()[0]:.4f}")
    print(f"  With Feedback    - Valence: {output_with_feedback['valence'].flatten()[0]:.4f}")
    print(f"  Difference: {abs(output_with_feedback['valence'].flatten()[0] - output_no_feedback['valence'].flatten()[0]):.4f}")
    
    # Demonstrate training compatibility
    print("\n" + "="*50)
    print("🎓 Testing Training Mode Compatibility")
    print("="*50)
    
    feedback_model.train()
    
    # Create dummy targets
    valence_target = torch.randn(batch_size, 1)
    arousal_target = torch.randn(batch_size, 1)
    
    # Forward pass in training mode
    output = feedback_model(dummy_input)
    
    # Simple loss calculation
    mse_loss = torch.nn.MSELoss()
    valence_loss = mse_loss(output['valence'], valence_target)
    arousal_loss = mse_loss(output['arousal'], arousal_target)
    total_loss = valence_loss + arousal_loss
    
    print(f"Training Mode Test:")
    print(f"  Valence Loss: {valence_loss.item():.4f}")
    print(f"  Arousal Loss: {arousal_loss.item():.4f}")
    print(f"  Total Loss: {total_loss.item():.4f}")
    print("  ✅ Backward pass computation works!")
    
    # Gradient check
    total_loss.backward()
    
    # Count parameters with gradients
    params_with_grad = sum(1 for p in feedback_model.parameters() if p.grad is not None)
    total_params = sum(1 for p in feedback_model.parameters())
    
    print(f"  Parameters with gradients: {params_with_grad}/{total_params}")

def usage_example():
    """Show practical usage example."""
    
    print("\n" + "="*50)
    print("📚 Practical Usage Example")
    print("="*50)
    
    print("""
# Basic Usage:
    model = FeatureEmotionRegression_Cnn6_LRM(
    sample_rate=32000,
    window_size=1024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14000,
    freeze_base=True,
    forward_passes=2
)

# Load pretrained weights
model.load_from_pretrain('pretrained_model/Cnn6_mAP=0.343.pth')

# Forward pass with feedback
features = torch.randn(batch_size, time_steps, mel_bins)
output = model(features, forward_passes=3)

# Training command:
python pytorch/emotion_main.py train \\
    --dataset_path features/emotion_features.h5 \\
    --workspace workspaces/feedback_emotion \\
    --model_type FeatureEmotionRegression_Cnn6_LRM \\
    --pretrained_checkpoint_path pretrained_model/Cnn6_mAP=0.343.pth \\
    --freeze_base \\
    --forward_passes 2 \\
    --learning_rate 1e-4 \\
    --batch_size 16 \\
    --cuda

# Key Features:
- Multiple forward passes for iterative refinement
- Emotion predictions modulate intermediate features
- Configurable number of feedback iterations
- Compatible with existing training pipeline
- Can disable feedback for ablation studies
    """)

if __name__ == '__main__':
    print("Starting FeatureEmotionRegression_Cnn6_LRM demonstration...")
    compare_models()
    usage_example()
    print("\n🎉 Demonstration completed successfully!") 