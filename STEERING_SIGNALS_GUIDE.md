# External Steering Signals for Emotion Feedback Testing

This guide explains how to create and use external steering signals for testing steering feedback in the emotion analysis system.

## Overview

External steering signals allow you to inject predefined feedback patterns into the emotion model to test how different emotional categories influence the model's predictions. The system categorizes audio data into 9 bins based on valence and arousal dimensions, then extracts average activations from each category to create steering signals.

## The 9 Emotion Categories

The system divides the 2D valence-arousal space into 9 categories:

| Valence \ Arousal | Weak | Middle | Strong |
|-------------------|------|--------|--------|
| **Negative** | negative_weak | negative_middle | negative_strong |
| **Neutral** | neutral_weak | neutral_middle | neutral_strong |
| **Positive** | positive_weak | positive_middle | positive_strong |

### Categorization Methods

1. **Quantile-based** (default): Splits each dimension into 3 equal groups using 33rd and 67th percentiles
2. **Fixed thresholds**: Uses predefined boundaries (valence: 0.3, 0.7; arousal: 0.3, 0.7)

## Quick Start

### 1. Generate Steering Signals

```bash
# Run the complete pipeline
bash shell_scripts/run_steering_signals.sh
```

Or run manually:

```bash
# Step 1: Generate steering signals
python scripts/generate_steering_signals.py \
    --dataset_path features/emotion_features.h5 \
    --model_checkpoint workspaces/emotion_feedback/checkpoints/.../best_model.pth \
    --output_dir steering_signals \
    --categorization_method quantile \
    --cuda

# Step 2: Test steering signals
python scripts/test_steering_signals.py \
    --model_checkpoint workspaces/emotion_feedback/checkpoints/.../best_model.pth \
    --steering_dir steering_signals \
    --dataset_path features/emotion_features.h5 \
    --output_dir steering_test_results \
    --num_samples 20 \
    --cuda
```

### 2. Examine Results

The pipeline generates:

- **Steering signals**: Average activations for each emotion category
- **Visualizations**: 2D scatter plots and distribution charts
- **Test results**: Statistical analysis of steering effects

## File Structure

```
steering_signals/
├── category_info.json              # Category definitions and boundaries
├── emotion_categories.png          # 2D scatter plot of categories
├── category_distribution.png       # Bar chart of category counts
├── steering_signals_summary.txt    # Summary of generated signals
├── negative_weak/                  # Steering signals for each category
│   ├── valence_128d.npy
│   ├── arousal_128d.npy
│   ├── valence_256d.npy
│   ├── arousal_256d.npy
│   ├── visual_embedding.npy
│   ├── valence_output.npy
│   └── arousal_output.npy
├── negative_middle/
│   └── ...
└── ... (7 more categories)

steering_test_results/
├── steering_test_results.json      # Detailed test results
├── steering_test_summary.txt       # Statistical summary
├── valence_changes.png             # Box plot of valence changes
├── arousal_changes.png             # Box plot of arousal changes
└── steering_effects_2d.png         # 2D scatter of steering effects
```

## Using Steering Signals in Your Code

### Load Steering Signals

```python
import numpy as np
import json
import torch

def load_steering_signals(steering_dir):
    """Load steering signals from directory."""
    # Load category information
    with open(os.path.join(steering_dir, "category_info.json"), 'r') as f:
        category_info = json.load(f)
    
    # Load steering signals
    steering_signals = {}
    for category_name in category_info['descriptions'].values():
        cat_dir = os.path.join(steering_dir, category_name)
        if os.path.exists(cat_dir):
            steering_signals[category_name] = {}
            for signal_file in os.listdir(cat_dir):
                if signal_file.endswith('.npy'):
                    signal_type = signal_file[:-4]
                    signal_path = os.path.join(cat_dir, signal_file)
                    signal_data = np.load(signal_path)
                    steering_signals[category_name][signal_type] = signal_data
    
    return steering_signals, category_info

# Load steering signals
steering_signals, category_info = load_steering_signals('steering_signals')
```

### Apply Steering to Model

```python
# Load your trained model
model = load_emotion_model('path/to/checkpoint.pth', device)

# Apply steering signals
category_name = 'positive_strong'  # Choose category
valence_128d = torch.from_numpy(steering_signals[category_name]['valence_128d']).float().to(device)
arousal_128d = torch.from_numpy(steering_signals[category_name]['arousal_128d']).float().to(device)

# Set external feedback
model.set_external_feedback(valence_128d=valence_128d, arousal_128d=arousal_128d)

# Run inference with steering
with torch.no_grad():
    output = model(input_features, forward_passes=2)
    steered_valence = output['valence'].cpu().numpy()
    steered_arousal = output['arousal'].cpu().numpy()

# Clear feedback state for next sample
model.clear_feedback_state()
```

### Compare with Baseline

```python
# Baseline prediction (no steering)
with torch.no_grad():
    baseline_output = model(input_features, forward_passes=1)
    baseline_valence = baseline_output['valence'].cpu().numpy()
    baseline_arousal = baseline_output['arousal'].cpu().numpy()

# Calculate steering effects
valence_change = steered_valence - baseline_valence
arousal_change = steered_arousal - baseline_arousal

print(f"Valence change: {valence_change:.4f}")
print(f"Arousal change: {arousal_change:.4f}")
```

## Advanced Usage

### Custom Categorization

You can modify the categorization logic in `scripts/generate_steering_signals.py`:

```python
def categorize_emotion_data(valence, arousal, method='quantile'):
    if method == 'custom':
        # Define your own boundaries
        category_info = {
            'valence_bounds': {
                'negative': (-np.inf, 0.2),
                'neutral': (0.2, 0.8),
                'positive': (0.8, np.inf)
            },
            'arousal_bounds': {
                'weak': (-np.inf, 0.2),
                'middle': (0.2, 0.8),
                'strong': (0.8, np.inf)
            }
        }
        # ... rest of categorization logic
```

### Multiple Steering Signals

Apply multiple steering signals with different strengths:

```python
# Apply steering with different strengths
for strength in [0.5, 1.0, 2.0]:
    valence_128d_scaled = valence_128d * strength
    arousal_128d_scaled = arousal_128d * strength
    
    model.set_external_feedback(
        valence_128d=valence_128d_scaled, 
        arousal_128d=arousal_128d_scaled
    )
    
    output = model(input_features, forward_passes=2)
    # Analyze effects...
```

### Steering Signal Analysis

Analyze the characteristics of steering signals:

```python
# Analyze steering signal statistics
for category_name, signals in steering_signals.items():
    valence_128d = signals['valence_128d']
    arousal_128d = signals['arousal_128d']
    
    print(f"\nCategory: {category_name}")
    print(f"  Valence 128D - Mean: {np.mean(valence_128d):.4f}, Std: {np.std(valence_128d):.4f}")
    print(f"  Arousal 128D - Mean: {np.mean(arousal_128d):.4f}, Std: {np.std(arousal_128d):.4f}")
    print(f"  Valence-Arousal correlation: {np.corrcoef(valence_128d, arousal_128d)[0,1]:.4f}")
```

## Understanding the Results

### Valence/Arousal Changes

The test results show how each steering signal affects predictions:

- **Positive changes**: Steering signal increases the predicted value
- **Negative changes**: Steering signal decreases the predicted value
- **Magnitude**: How much the steering signal influences predictions

### Category Effectiveness

Different categories may have varying effects:

- **Strong categories** (high arousal/valence): Often produce larger changes
- **Weak categories** (low arousal/valence): May produce smaller or more subtle changes
- **Neutral categories**: May act as "baseline" or "calming" signals

### Consistency Analysis

Look for patterns in the results:

- **Consistent effects**: Same category always produces similar changes
- **Sample-dependent effects**: Effects vary based on input characteristics
- **Interaction effects**: Combinations of valence/arousal steering

## Troubleshooting

### Common Issues

1. **Model loading errors**: Ensure checkpoint path is correct and model architecture matches
2. **Memory issues**: Reduce batch size or number of samples
3. **No steering effects**: Check that steering signals are properly loaded and applied
4. **Inconsistent results**: Verify that feedback state is cleared between samples

### Debugging Tips

```python
# Check steering signal shapes
print(f"Valence 128D shape: {valence_128d.shape}")
print(f"Arousal 128D shape: {arousal_128d.shape}")

# Verify model feedback state
print(f"Model has feedback: {hasattr(model, 'valence_128d')}")
print(f"Valence 128D stored: {model.valence_128d is not None}")

# Check input features
print(f"Input shape: {input_features.shape}")
print(f"Input range: [{input_features.min():.3f}, {input_features.max():.3f}]")
```

## Research Applications

### Emotion Manipulation Studies

- Test how different emotional contexts affect perception
- Study the relationship between valence and arousal
- Investigate emotional priming effects

### Model Interpretability

- Understand which features contribute to emotion predictions
- Analyze the internal representations of different emotions
- Study the feedback mechanisms in the model

### Adaptive Systems

- Develop emotion-aware audio processing systems
- Create personalized emotion recognition models
- Build interactive emotion feedback systems

## References

- Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology*, 39(6), 1161-1178.
- Long Range Modulation (LRM) feedback mechanisms
- Emotion recognition in audio signals
- External feedback in neural networks

## Contributing

To extend the steering signals system:

1. Add new categorization methods
2. Implement different steering signal types
3. Create additional analysis tools
4. Develop new visualization methods

Submit pull requests with clear documentation and examples. 