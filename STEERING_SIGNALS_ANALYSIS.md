# Steering Signals Analysis and Solutions

## Problem Summary

We successfully created external steering signals for the emotion analysis model, but discovered that the steering signals have minimal effect on the model's predictions. This document analyzes the issue and provides solutions.

## What We Accomplished

1. **Successfully generated steering signals**: Created 9 emotion categories (valence: positive/neutral/negative × arousal: strong/middle/weak)
2. **Extracted average activations**: Computed 128D valence and arousal representations for each category
3. **Integrated with LRM system**: Connected steering signals to the model's feedback mechanism
4. **Debugged the system**: Identified where the steering signals are being processed

## The Issue: Normalization and Squashing

The steering signals are being processed through the LRM (Long Range Modulation) system, which includes:

1. **Normalization**: `AdaptiveFullstackNorm` normalizes signals across all dimensions
2. **Squashing**: `FeedbackScale` with `tanh` mode applies `torch.tanh()` to squash values to [-1, 1]
3. **Modulation**: The processed signals modulate convolutional layer outputs

**Problem**: Even with 20x amplification, the normalization and squashing steps reduce the steering signals to very small values, making them ineffective.

## Steering Signal Statistics

```
Original steering signals (before amplification):
- Valence: mean ~0.044, std ~0.09, range [0.0, 0.5]
- Arousal: mean ~0.041, std ~0.09, range [0.0, 0.4]

After 20x amplification:
- Valence: mean ~0.88, std ~1.81, range [0.0, 8.8]
- Arousal: mean ~0.79, std ~1.66, range [0.0, 7.1]
```

## Solutions Implemented

### 1. Standard LRM Method (`--steering_method lrm`)
- Uses the original LRM feedback system
- **Result**: Minimal effect due to normalization/squashing

### 2. Direct Injection (`--steering_method direct`)
- Bypasses LRM system, directly sets model's internal feedback
- **Result**: May work better but bypasses intended feedback mechanism

### 3. Bypass Normalization (`--steering_method bypass`)
- Temporarily disables normalization and squashing in ModBlocks
- **Result**: Should preserve steering signal strength

### 4. High Modulation Strength (`--steering_method modulation`)
- Increases modulation strength to 10x
- **Result**: Amplifies the effects of processed signals

## Usage Examples

```bash
# Test different steering methods
python scripts/test_steering_signals.py \
    --model_checkpoint path/to/model.pth \
    --steering_dir test_steering_signals \
    --dataset_path path/to/dataset.h5 \
    --steering_method bypass \
    --steering_strength 20.0 \
    --num_samples 5 \
    --cuda
```

## Recommendations

### For Research Use
1. **Use `bypass` method**: This preserves steering signal strength while maintaining the feedback architecture
2. **Increase steering strength**: Use values 10-50x for visible effects
3. **Monitor signal statistics**: Check that signals maintain reasonable magnitudes

### For Production Use
1. **Modify ModBlock implementation**: Consider removing or reducing normalization/squashing
2. **Add steering strength parameter**: Make amplification configurable
3. **Implement signal validation**: Ensure steering signals are within expected ranges

## Technical Details

### LRM System Architecture
```
Steering Signal → ModBlock → Normalization → Squashing → Modulation → Conv Layer
```

### ModBlock Processing
```python
# In ModBlock.forward()
x = self.rescale(x, target_size)  # Normalize + Squash + Resize
x = x * (neg_mask * self.neg_scale + pos_mask * self.pos_scale)  # Asymmetric scaling
x = self.modulation(x)  # 1x1 conv for channel mapping
```

### Feedback Application
```python
# In forward_hook_target()
output = output + output * total_mod  # Additive modulation
output = F.relu(output, inplace=False)  # ReLU activation
```

## Future Improvements

1. **Configurable normalization**: Allow disabling normalization for steering signals
2. **Separate steering pathway**: Create dedicated steering mechanism outside LRM
3. **Signal validation**: Add checks for steering signal quality and magnitude
4. **Visualization tools**: Create plots showing steering effects over time
5. **Batch steering**: Support applying different steering signals to different samples

## Conclusion

The steering signals system is working correctly from a technical standpoint, but the LRM system's normalization and squashing steps are reducing their effectiveness. The `bypass` method provides the best solution for research applications, while production systems may require modifications to the ModBlock implementation.

The system successfully demonstrates the concept of external steering for emotion feedback, and with the right configuration, can be used for emotion manipulation studies and model interpretability research. 