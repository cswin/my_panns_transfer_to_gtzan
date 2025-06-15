# Emo-Soundscapes Data Setup Guide

This guide explains how to prepare the Emo-Soundscapes dataset for emotion regression training.

## Dataset Structure

Your data should be organized as:
```
/DATA/pliu/EmotionData/Emo-Soundscapes/
├── Emo-Soundscapes-Audio/
│   ├── 600_Sounds/
│   │   ├── subfolder1/
│   │   ├── subfolder2/
│   │   └── ...
│   └── 613_MixedSounds/
│       ├── subfolder1/
│       ├── subfolder2/
│       └── ...
└── Emo-Soundscapes-Ratings/
    ├── Valence.csv
    └── Arousal.csv
```

## Option 1: Flatten Directory Structure (Recommended)

This approach copies all audio files into a single directory for easier processing.

### Step 1: First, do a dry run to see what would be copied:

```bash
python flatten_audio_directory.py \
    --source_base /DATA/pliu/EmotionData/Emo-Soundscapes/Emo-Soundscapes-Audio \
    --output_dir /DATA/pliu/EmotionData/Emo-Soundscapes/audio_flat \
    --ratings_dir /DATA/pliu/EmotionData/Emo-Soundscapes/Emo-Soundscapes-Ratings
```

### Step 2: If the dry run looks good, actually copy the files:

```bash
python flatten_audio_directory.py \
    --source_base /DATA/pliu/EmotionData/Emo-Soundscapes/Emo-Soundscapes-Audio \
    --output_dir /DATA/pliu/EmotionData/Emo-Soundscapes/audio_flat \
    --ratings_dir /DATA/pliu/EmotionData/Emo-Soundscapes/Emo-Soundscapes-Ratings \
    --copy
```

### Step 3: Extract features from the flattened directory:

```bash
python extract_emotion_features.py \
    --audio_dir /DATA/pliu/EmotionData/Emo-Soundscapes/audio_flat \
    --ratings_dir /DATA/pliu/EmotionData/Emo-Soundscapes/Emo-Soundscapes-Ratings \
    --output_dir features/emotion_features
```

## Option 2: Direct Processing (No Copying)

This approach processes the nested directories directly without copying files.

### Extract features directly from nested directories:

```bash
python extract_emotion_features.py \
    --audio_dir /DATA/pliu/EmotionData/Emo-Soundscapes/Emo-Soundscapes-Audio/600_Sounds \
                 /DATA/pliu/EmotionData/Emo-Soundscapes/Emo-Soundscapes-Audio/613_MixedSounds \
    --ratings_dir /DATA/pliu/EmotionData/Emo-Soundscapes/Emo-Soundscapes-Ratings \
    --output_dir features/emotion_features
```

## Training the Model

Once features are extracted, use the training pipeline:

```bash
# Using the automated script (Cnn6 by default)
bash run_emotion.sh

# Or manually with specific parameters
python pytorch/emotion_main.py train \
    --feature_path features/emotion_features/emotion_features.h5 \
    --model_type FeatureEmotionRegression_Cnn6 \
    --loss_type mse \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --cuda
```

## Troubleshooting

### If you get CSV parsing errors:
The script now includes robust CSV parsing that handles malformed CSV files automatically.

### If audio files are not found:
1. Check that the directory paths are correct
2. Use the dry run option to see what files are found
3. Check the file extensions (script supports .wav, .mp3, .flac, .m4a, .aac)

### If ratings coverage is low:
The script will show you which files have ratings but no audio, and vice versa. Use this to debug filename mismatches.

## Expected Results

- Total audio files: ~1,213 (600 + 613)
- Files with both valence and arousal ratings: Should match the number in CSV files
- Final features: Shape will be [N, 64, 501] for mel-spectrograms
- Valence range: Typically [-1, 1]
- Arousal range: Typically [-1, 1] 