# Full-Length Audio Processing Update

This document summarizes the changes made to update the PANNs emotion regression system from segment-based processing to full-length audio processing.

## Overview

The system has been updated to process full-length audios instead of dividing them into segments. This change affects both the standard emotion regression model and the LRM feedback model.

## Key Changes Made

### 1. Feature Extraction (`scripts/extract_features.py`)

**Before (Segment-based):**
- Extracted 6 segments of 1 second each from 6-second audio clips
- Each audio file produced 6 separate samples
- Used `extract_melspectrogram_segments()` function

**After (Full-length):**
- Extracts mel-spectrograms from full audio files (up to 30 seconds)
- Each audio file produces 1 sample
- Uses `extract_melspectrogram_full_audio()` function
- Removed segment-related code and logic

### 2. Data Generator (`src/data/data_generator.py`)

**Before (Segment-based):**
- Included segment indices in data dictionaries
- Complex audio-based splitting logic to prevent data leakage
- Segment-aware collate functions

**After (Full-length):**
- Removed segment indices from data structures
- Simplified data splitting (direct index-based split)
- Updated collate functions to handle audio-level data
- Cleaner, more straightforward data loading

### 3. Emotion Models (`src/models/emotion_models.py`)

**Before (Segment-based):**
- Feedback signals computed for each segment
- Different feedback for each forward pass

**After (Full-length):**
- Feedback signals computed once per audio file
- Consistent feedback signals across multiple forward passes
- Updated forward method to handle full-length audio processing
- Ensures consistent modulation since there are no segments

### 4. Shell Scripts

#### `shell_scripts/run_emotion.sh`
- Updated comments to reflect full-length audio processing
- Reduced batch size from 32 to 16 (memory considerations for full-length audios)
- Updated iteration calculations for new dataset size
- Removed segment-related warnings and comments
- Simplified evaluation process

#### `shell_scripts/run_emotion_feedback.sh`
- Updated comments to explain full-length audio feedback mechanism
- Reduced batch size from 32 to 16
- Added explanation of consistent feedback signals across passes
- Updated training configuration for full-length audios

### 5. Test Files (`tests/test_data_split.py`)

**Before (Segment-based):**
- Complex logic to handle segment suffixes (`_seg0`, `_seg1`, etc.)
- Segment counting and validation
- Audio file deduplication logic

**After (Full-length):**
- Simplified data split validation
- Direct audio file comparison
- Removed segment-related logic
- Cleaner test output

## Technical Details

### Feedback Mechanism Changes

For the LRM feedback model, the key change is in how feedback signals are handled:

```python
# Before (segment-based)
for pass_idx in range(num_passes):
    # Compute feedback signals for each pass
    self._store_feedback_signals(valence_128d, arousal_128d)

# After (full-length audio)
feedback_computed = False
stored_valence_128d = None
stored_arousal_128d = None

for pass_idx in range(num_passes):
    if not feedback_computed:
        # Compute feedback signals once
        stored_valence_128d = valence_128d
        stored_arousal_128d = arousal_128d
        feedback_computed = True
    else:
        # Use stored feedback signals for consistency
        self.valence_128d = stored_valence_128d
        self.arousal_128d = stored_arousal_128d
```

### Data Structure Changes

**Before:**
- Features: `(num_segments * num_audio_files, time_steps, mel_bins)`
- Each audio file had multiple segment entries

**After:**
- Features: `(num_audio_files, time_steps, mel_bins)`
- Each audio file has one entry

### Memory Considerations

- Reduced batch size from 32 to 16 to accommodate full-length audio features
- Full-length audios can be significantly larger than 1-second segments
- Memory usage scales with audio duration

## Benefits of Full-Length Audio Processing

1. **Simplified Architecture**: No need to handle segment-level processing
2. **Consistent Feedback**: LRM feedback signals are consistent across passes
3. **Cleaner Data Flow**: Direct audio-to-feature mapping
4. **Better Temporal Context**: Full audio context available for emotion prediction
5. **Reduced Complexity**: Eliminates segment management overhead

## Usage

The updated scripts work exactly the same way as before:

```bash
# Standard emotion regression
bash shell_scripts/run_emotion.sh

# LRM feedback emotion regression
bash shell_scripts/run_emotion_feedback.sh
```

## Migration Notes

- Existing segment-based features will not work with the updated code
- New feature extraction is required for full-length audio processing
- The system now expects ~1200 audio files instead of ~7200 segment samples
- Batch sizes have been optimized for full-length audio memory requirements

## File Summary

Updated files:
- `scripts/extract_features.py` - Feature extraction for full-length audios
- `src/data/data_generator.py` - Data loading without segments
- `src/models/emotion_models.py` - Consistent feedback for full-length audios
- `shell_scripts/run_emotion.sh` - Updated for full-length processing
- `shell_scripts/run_emotion_feedback.sh` - Updated for full-length feedback
- `tests/test_data_split.py` - Simplified data split validation

The system is now optimized for full-length audio processing while maintaining all the advanced features of the LRM feedback mechanism. 