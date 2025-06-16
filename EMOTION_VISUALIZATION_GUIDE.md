# Emotion Prediction Visualization Guide

This guide explains how to use the enhanced emotion evaluation system that generates CSV files and comprehensive visualizations for emotion prediction results.

## Overview

The enhanced system now provides:

1. **CSV Export**: Saves predictions and ground truth values for both segment-level and audio-level analysis
2. **Scatter Plots**: True vs predicted plots with fit lines, correlation coefficients, and R² values
3. **Time-Series Plots**: Shows how predicted emotions change over time for each audio file
4. **Summary Statistics**: Error distributions, performance by emotion range, and correlation matrices

## Generated Files

When you run the emotion evaluation, the following files are automatically created:

### CSV Files

- **`segment_predictions.csv`**: Contains predictions for each audio segment
  - `audio_name`: Full segment name (e.g., "audio123_seg0")
  - `base_audio`: Base audio file name (e.g., "audio123")
  - `segment_idx`: Segment index within the audio (0, 1, 2, ...)
  - `valence_true`: Ground truth valence value
  - `valence_pred`: Predicted valence value
  - `arousal_true`: Ground truth arousal value
  - `arousal_pred`: Predicted arousal value

- **`audio_predictions.csv`**: Contains aggregated predictions for each audio file
  - `audio_name`: Audio file name
  - `valence_true`: Ground truth valence (same for all segments)
  - `valence_pred`: Mean predicted valence across segments
  - `arousal_true`: Ground truth arousal (same for all segments)
  - `arousal_pred`: Mean predicted arousal across segments
  - `num_segments`: Number of segments for this audio file

### Visualization Files

#### Main Plots Directory (`plots/`)

1. **`audio_scatter_plots.png/pdf`**: Scatter plots for audio-level predictions
   - True vs predicted valence and arousal
   - Fit lines with slope and intercept
   - Perfect prediction line (y=x)
   - Statistics: Pearson r, R², RMSE, MAE, p-value

2. **`segment_scatter_plots.png/pdf`**: Scatter plots for segment-level predictions
   - Same format as audio-level but with all segments included

3. **`time_series_sample.png/pdf`**: Grid showing time-series for first 12 audio files
   - Shows how valence and arousal change over time (segments)
   - Compares true vs predicted values

4. **`summary_statistics.png/pdf`**: Comprehensive summary plots including:
   - Error distribution histograms
   - Value distribution comparisons
   - Performance by emotion range
   - Correlation matrix

#### Individual Time-Series Directory (`plots/individual_timeseries/`)

Contains individual time-series plots for each audio file showing:
- Valence over time (true vs predicted)
- Arousal over time (true vs predicted)
- Filled areas showing prediction errors

## Usage

### Automatic Generation (Recommended)

The easiest way is to run the main evaluation script:

```bash
# Run the full pipeline with visualization
bash run_emotion.sh

# Or run inference on existing features
bash run_emotion.sh --skip-extraction
```

This automatically:
1. Evaluates the trained model
2. Saves predictions to CSV files
3. Generates all visualizations
4. Prints file locations

### Manual Generation

If you already have a trained model and want to generate visualizations:

```bash
# Run inference with CSV export
python pytorch/emotion_main.py inference \
    --model_path /path/to/model.pth \
    --dataset_path /path/to/features.h5 \
    --model_type "FeatureEmotionRegression_Cnn6" \
    --batch_size 32 \
    --cuda
```

### Generate Plots from Existing CSV Files

If you have CSV files and want to regenerate visualizations:

```bash
# Generate plots from existing CSV files
python generate_emotion_plots.py /path/to/predictions/directory
```

## Interpreting the Results

### Scatter Plots

- **Points close to the diagonal line (y=x)**: Good predictions
- **Points scattered widely**: Poor predictions
- **R² close to 1**: Strong linear relationship
- **Pearson r close to ±1**: Strong correlation

### Time-Series Plots

- **Smooth curves**: Consistent predictions over time
- **Large gaps between true/predicted**: Systematic errors
- **Similar trends**: Model captures temporal patterns
- **Opposite trends**: Model misses temporal dynamics

### Summary Statistics

- **Error distributions centered at 0**: Unbiased predictions
- **Narrow error distributions**: Consistent performance
- **Performance by emotion range**: Shows if model works better for certain emotion levels

## File Locations

After running evaluation, files are saved in:
```
workspaces/emotion_regression/checkpoints/.../predictions/
├── segment_predictions.csv
├── audio_predictions.csv
└── plots/
    ├── audio_scatter_plots.png
    ├── segment_scatter_plots.png
    ├── time_series_sample.png
    ├── summary_statistics.png
    └── individual_timeseries/
        ├── audio_file_1.png
        ├── audio_file_2.png
        └── ...
```

## Dependencies

Make sure you have all required packages installed:

```bash
pip install matplotlib seaborn pandas numpy scipy scikit-learn
```

## Customization

To customize the visualizations, edit `emotion_visualize.py`:

- Change plot colors by modifying the color palettes
- Adjust figure sizes by changing `figsize` parameters
- Add new metrics by extending the statistics calculation functions
- Modify plot styles by changing matplotlib/seaborn settings

## Troubleshooting

### Common Issues

1. **"emotion_visualize module not found"**: Make sure the script is in the same directory
2. **Missing CSV files**: Check that the evaluation ran successfully
3. **Empty plots**: Verify that the CSV files contain data
4. **Memory errors**: Reduce the number of individual time-series plots or increase batch size

### Manual Plot Generation

If automatic generation fails, you can manually generate plots:

```python
from emotion_visualize import create_emotion_visualizations
create_emotion_visualizations('/path/to/predictions')
```

## Example Analysis Workflow

1. **Train the model**: `bash run_emotion.sh`
2. **Check overall performance**: Look at `audio_scatter_plots.png`
3. **Analyze temporal patterns**: Review `time_series_sample.png`
4. **Investigate specific audios**: Browse `individual_timeseries/` folder
5. **Understand errors**: Study `summary_statistics.png`
6. **Export for further analysis**: Use the CSV files in your own scripts

This enhanced system provides comprehensive insights into your emotion prediction model's performance at both segment and audio levels, enabling better model analysis and improvement. 