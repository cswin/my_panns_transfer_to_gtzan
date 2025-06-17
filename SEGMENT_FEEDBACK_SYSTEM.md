# Segment-Based Feedback System for Audio Emotion Prediction

## Overview

This document describes the segment-based feedback system implemented for Long Range Modulation (LRM) in audio emotion prediction. The system enables **temporal feedback** where emotion predictions from one audio segment influence the processing of subsequent segments from the same audio file.

## Problem Statement

### Traditional Approach
- Each 1-second audio segment processed **independently**
- No temporal continuity between segments
- Final prediction = average of all segment predictions
- Segments processed in **random batch order**

### LRM Feedback Approach  
- Segments from same audio file processed **sequentially**
- Emotion predictions from segment N **modulate** segment N+1
- Maintains temporal continuity within each audio file
- Enables **dynamic adaptation** as the model "listens" to the audio

## System Architecture

### 1. Data Processing Flow

```
Audio File (6 seconds) → 6 segments (1 second each)
├── Segment 0: Pure feedforward (no feedback)
├── Segment 1: Modulated by feedback from Segment 0
├── Segment 2: Modulated by feedback from Segment 1
├── Segment 3: Modulated by feedback from Segment 2
├── Segment 4: Modulated by feedback from Segment 3
└── Segment 5: Modulated by feedback from Segment 4

Final Prediction = Average of all 6 segment predictions
```

### 2. Feedback Mechanism

**Source**: Affective system outputs (128-dimensional embeddings)
- Valence pathway: 512→256→128→1 (128D used for feedback)
- Arousal pathway: 512→256→128→1 (128D used for feedback)

**Target**: Visual system conv layers
- Valence → conv3, conv4 (semantic processing)
- Arousal → conv1, conv2 (attention to details)

**Process**:
1. Process segment N → extract 128D valence/arousal embeddings
2. Store embeddings as modulation signals
3. Process segment N+1 → conv layers modulated by stored signals
4. Repeat for all segments in sequence

### 3. Psychological Motivation

**Valence Feedback** (positive/negative emotion):
- Targets higher-level conv layers (conv3, conv4)
- Affects **semantic interpretation** of audio content
- Example: Positive valence enhances detection of pleasant sounds

**Arousal Feedback** (energy/activation level):
- Targets lower-level conv layers (conv1, conv2)  
- Affects **attention to acoustic details**
- Example: High arousal increases sensitivity to dynamic changes

## Implementation Details

### 1. LRM Model (`FeatureEmotionRegression_Cnn6_LRM`)

```python
# Separate affective pathways for rich feedback
self.embedding_valence_transform = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.Tanh()
)

self.embedding_arousal_transform = nn.Sequential(
    nn.Linear(512, 256), nn.ReLU(), 
    nn.Linear(256, 128), nn.Tanh()
)

# Psychologically-motivated connections
mod_connections = [
    {'source': 'embedding_valence', 'target': 'base.conv_block4'},
    {'source': 'embedding_valence', 'target': 'base.conv_block3'},
    {'source': 'embedding_arousal', 'target': 'base.conv_block2'},
    {'source': 'embedding_arousal', 'target': 'base.conv_block1'},
]
```

### 2. LRM Evaluator (`LRMEmotionEvaluator`)

**Key Features**:
- Groups segments by base audio file
- Processes segments sequentially within each audio file
- Maintains feedback state between segments
- Clears feedback state between different audio files
- Provides detailed feedback analysis

**Processing Flow**:
```python
for audio_file in audio_files:
    model.lrm.clear_stored_activations()  # Reset for new audio
    
    for segment in audio_file.segments:
        if segment_idx == 0:
            # First segment: pure feedforward
            output = model(segment, forward_passes=1)
        else:
            # Subsequent segments: use feedback from previous
            output = model(segment, forward_passes=1)
        
        # Store feedback for next segment
        if not last_segment:
            store_modulation_signals(output['embedding'])
```

### 3. Feedback Analysis

The system provides detailed analysis of feedback effects:

**Segment Position Effects**:
- Performance metrics by segment position (0, 1, 2, 3, 4, 5)
- Shows how feedback improves/changes predictions over time

**Prediction Consistency**:
- Measures variability of predictions within each audio file
- Lower variability may indicate better temporal modeling

**Example Output**:
```
=== LRM FEEDBACK ANALYSIS ===
Number of audio files: 1213
Average segments per audio: 6.0

--- Performance by Segment Position ---
segment_0: Count=1213, Valence MAE=0.1234, Arousal MAE=0.1456
segment_1: Count=1213, Valence MAE=0.1198, Arousal MAE=0.1423
segment_2: Count=1213, Valence MAE=0.1187, Arousal MAE=0.1401
...

--- Prediction Consistency Within Audio Files ---
Valence std (mean±std): 0.0456±0.0123
Arousal std (mean±std): 0.0523±0.0145
```

## Usage

### 1. Training with Segment Feedback

```bash
# Use the feedback training script
bash run_emotion_feedback.sh

# Or manually:
python pytorch/emotion_main.py train \
    --model_type "FeatureEmotionRegression_Cnn6_LRM" \
    --forward_passes 2 \
    --batch_size 16 \
    [other args...]
```

### 2. Evaluation with Feedback Analysis

```python
from emotion_evaluate_lrm import LRMEmotionEvaluator

evaluator = LRMEmotionEvaluator(model=lrm_model)
statistics, feedback_analysis = evaluator.evaluate_with_feedback_analysis(
    data_loader, save_predictions=True, output_dir="predictions"
)

evaluator.print_evaluation(statistics)
evaluator.print_feedback_analysis(feedback_analysis)
```

### 3. Testing the System

```bash
# Test segment-based feedback functionality
python test_segment_feedback.py
```

## Expected Benefits

### 1. Temporal Consistency
- Predictions should be more consistent within each audio file
- Reduced prediction variance across segments

### 2. Contextual Adaptation
- Model adapts its processing based on previous segments
- Better handling of temporal dynamics in emotion

### 3. Improved Performance
- Potentially better audio-level aggregated metrics
- More robust predictions for longer audio sequences

## Key Differences from Standard Approach

| Aspect | Standard Model | LRM Feedback Model |
|--------|---------------|-------------------|
| **Segment Processing** | Independent, random order | Sequential, by audio file |
| **Temporal Context** | None | Previous segments influence current |
| **Feedback Source** | N/A | 128D affective embeddings |
| **Feedback Target** | N/A | Conv layer features |
| **Evaluation** | Standard evaluator | LRM-aware evaluator |
| **Analysis** | Basic metrics | Detailed feedback analysis |

## Files Modified/Created

### Core Implementation
- `pytorch/models_lrm.py` - LRM model with segment feedback
- `pytorch/emotion_evaluate_lrm.py` - LRM-aware evaluator
- `pytorch/emotion_main.py` - Updated to use LRM evaluator

### Testing & Scripts
- `test_segment_feedback.py` - Comprehensive testing
- `run_emotion_feedback.sh` - Training script for LRM models

### Documentation
- `SEGMENT_FEEDBACK_SYSTEM.md` - This document
- `LRM_MODEL_ARCHITECTURE.md` - Detailed architecture docs

## Future Enhancements

### 1. Multi-Pass Feedback
- Currently: 1 pass per segment with inter-segment feedback
- Future: Multiple passes per segment + inter-segment feedback

### 2. Bidirectional Feedback
- Currently: Forward-only (segment N → segment N+1)
- Future: Bidirectional (segment N ↔ segment N+1)

### 3. Attention-Based Feedback
- Currently: Fixed modulation connections
- Future: Learned attention weights for dynamic connections

### 4. Cross-Audio Feedback
- Currently: Feedback only within same audio file
- Future: Feedback across similar audio files in batch

## Conclusion

The segment-based feedback system provides a novel approach to audio emotion prediction by incorporating temporal dynamics through LRM. The system maintains the benefits of segment-based processing while adding contextual awareness that can improve prediction quality and consistency.

The implementation is fully integrated with the existing training and evaluation pipeline, making it easy to compare with standard approaches and analyze the effects of feedback on model performance. 