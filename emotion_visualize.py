#!/usr/bin/env python3
"""
Visualization script for emotion prediction results.

This script creates:
1. Scatter plots with fit lines and R² values for true vs predicted emotions
2. Time-series plots showing predicted emotions over time for each audio file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr
import os
import argparse
from pathlib import Path


def create_emotion_visualizations(prediction_dir):
    """Create all emotion visualizations from prediction CSV files.
    
    Args:
        prediction_dir: Directory containing segment_predictions.csv and audio_predictions.csv
    """
    prediction_dir = Path(prediction_dir)
    
    # Load data
    segment_df = pd.read_csv(prediction_dir / 'segment_predictions.csv')
    audio_df = pd.read_csv(prediction_dir / 'audio_predictions.csv')
    
    # Create output directory for plots
    plots_dir = prediction_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print(f"Creating visualizations in: {plots_dir}")
    
    # 1. Create scatter plots for audio-level predictions
    create_scatter_plots(audio_df, plots_dir, level='audio')
    
    # 2. Create scatter plots for segment-level predictions  
    create_scatter_plots(segment_df, plots_dir, level='segment')
    
    # 3. Create time-series plots for individual audio files
    create_time_series_plots(segment_df, plots_dir)
    
    # 4. Create summary statistics plots
    create_summary_plots(audio_df, segment_df, plots_dir)
    
    print(f"All visualizations saved in: {plots_dir}")


def create_scatter_plots(df, plots_dir, level='audio'):
    """Create scatter plots with fit lines for true vs predicted values.
    
    Args:
        df: DataFrame with predictions
        plots_dir: Directory to save plots
        level: 'audio' or 'segment' level
    """
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{level.title()}-Level Emotion Predictions: True vs Predicted', fontsize=16, fontweight='bold')
    
    emotions = ['valence', 'arousal']
    colors = ['#FF6B6B', '#4ECDC4']
    
    for idx, emotion in enumerate(emotions):
        ax = axes[idx]
        
        true_col = f'{emotion}_true'
        pred_col = f'{emotion}_pred'
        
        x = df[true_col]
        y = df[pred_col]
        
        # Create scatter plot
        ax.scatter(x, y, alpha=0.6, color=colors[idx], s=50, edgecolors='white', linewidth=0.5)
        
        # Calculate fit line
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        line_x = np.linspace(x.min(), x.max(), 100)
        line_y = slope * line_x + intercept
        
        # Plot fit line
        ax.plot(line_x, line_y, color='red', linewidth=2, linestyle='--', alpha=0.8, label=f'Fit line')
        
        # Plot perfect prediction line (y=x)
        ax.plot([x.min(), x.max()], [x.min(), x.max()], 
                color='black', linewidth=1, linestyle='-', alpha=0.5, label='Perfect prediction')
        
        # Calculate metrics
        pearson_r, pearson_p = pearsonr(x, y)
        rmse = np.sqrt(np.mean((x - y) ** 2))
        mae = np.mean(np.abs(x - y))
        
        # Formatting
        ax.set_xlabel(f'True {emotion.title()}', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'Predicted {emotion.title()}', fontsize=12, fontweight='bold')
        ax.set_title(f'{emotion.title()} Prediction', fontsize=14, fontweight='bold')
        
        # Add statistics text
        stats_text = f'r = {pearson_r:.3f}\nR² = {r_value**2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}\np = {pearson_p:.3e}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        # Set equal aspect ratio and limits
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), 
                max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(plots_dir / f'{level}_scatter_plots.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / f'{level}_scatter_plots.pdf', bbox_inches='tight')
    plt.close()
    print(f"✓ {level.title()}-level scatter plots saved")


def create_time_series_plots(segment_df, plots_dir):
    """Create time-series plots showing predictions over time for each audio.
    
    Args:
        segment_df: DataFrame with segment-level predictions
        plots_dir: Directory to save plots
    """
    # Group by base audio
    audio_groups = segment_df.groupby('base_audio')
    
    # Create individual plots for first few audio files (to avoid too many plots)
    sample_audios = list(audio_groups.groups.keys())[:12]  # Show first 12 audio files
    
    # Create a grid of subplots
    n_cols = 3
    n_rows = 4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 20))
    fig.suptitle('Emotion Predictions Over Time (First 12 Audio Files)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    for idx, audio_name in enumerate(sample_audios):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        audio_data = audio_groups.get_group(audio_name).sort_values('segment_idx')
        
        # Time axis (segment indices)
        time_points = audio_data['segment_idx']
        
        # Plot valence and arousal
        ax.plot(time_points, audio_data['valence_true'], 'o-', color='#FF6B6B', 
                linewidth=2, markersize=4, label='True Valence', alpha=0.7)
        ax.plot(time_points, audio_data['valence_pred'], 's--', color='#FF6B6B', 
                linewidth=1.5, markersize=3, label='Pred Valence', alpha=0.9)
        
        ax.plot(time_points, audio_data['arousal_true'], 'o-', color='#4ECDC4', 
                linewidth=2, markersize=4, label='True Arousal', alpha=0.7)
        ax.plot(time_points, audio_data['arousal_pred'], 's--', color='#4ECDC4', 
                linewidth=1.5, markersize=3, label='Pred Arousal', alpha=0.9)
        
        ax.set_title(f'{audio_name[:30]}{"..." if len(audio_name) > 30 else ""}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Segment Index', fontsize=9)
        ax.set_ylabel('Emotion Value', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(-1.1, 1.1)
    
    # Hide unused subplots
    for idx in range(len(sample_audios), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'time_series_sample.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'time_series_sample.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Time-series sample plots saved")
    
    # Create individual plots for each audio (saved separately)
    timeseries_dir = plots_dir / 'individual_timeseries'
    timeseries_dir.mkdir(exist_ok=True)
    
    print(f"Creating individual time-series plots for {len(audio_groups)} audio files...")
    for audio_name, audio_data in audio_groups:
        create_individual_timeseries(audio_data, audio_name, timeseries_dir)
    
    print(f"✓ Individual time-series plots saved in: {timeseries_dir}")


def create_individual_timeseries(audio_data, audio_name, output_dir):
    """Create individual time-series plot for one audio file.
    
    Args:
        audio_data: DataFrame with data for one audio file
        audio_name: Name of the audio file
        output_dir: Directory to save the plot
    """
    audio_data = audio_data.sort_values('segment_idx')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(f'Emotion Predictions: {audio_name}', fontsize=14, fontweight='bold')
    
    time_points = audio_data['segment_idx']
    
    # Valence plot
    ax1.plot(time_points, audio_data['valence_true'], 'o-', color='#FF6B6B', 
             linewidth=2, markersize=6, label='True Valence', alpha=0.7)
    ax1.plot(time_points, audio_data['valence_pred'], 's--', color='#FF4444', 
             linewidth=2, markersize=4, label='Predicted Valence')
    ax1.fill_between(time_points, audio_data['valence_true'], audio_data['valence_pred'], 
                     alpha=0.2, color='#FF6B6B')
    ax1.set_ylabel('Valence', fontsize=12, fontweight='bold')
    ax1.set_title('Valence Over Time', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(-1.1, 1.1)
    
    # Arousal plot
    ax2.plot(time_points, audio_data['arousal_true'], 'o-', color='#4ECDC4', 
             linewidth=2, markersize=6, label='True Arousal', alpha=0.7)
    ax2.plot(time_points, audio_data['arousal_pred'], 's--', color='#2E8B8B', 
             linewidth=2, markersize=4, label='Predicted Arousal')
    ax2.fill_between(time_points, audio_data['arousal_true'], audio_data['arousal_pred'], 
                     alpha=0.2, color='#4ECDC4')
    ax2.set_xlabel('Segment Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Arousal', fontsize=12, fontweight='bold')
    ax2.set_title('Arousal Over Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    # Clean filename
    safe_filename = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in audio_name)
    safe_filename = safe_filename.replace(' ', '_')[:50]  # Limit length
    
    plt.savefig(output_dir / f'{safe_filename}.png', dpi=200, bbox_inches='tight')
    plt.close()


def create_summary_plots(audio_df, segment_df, plots_dir):
    """Create summary statistics and distribution plots.
    
    Args:
        audio_df: DataFrame with audio-level predictions
        segment_df: DataFrame with segment-level predictions
        plots_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Emotion Prediction Summary Statistics', fontsize=16, fontweight='bold')
    
    # Error distributions
    audio_df['valence_error'] = audio_df['valence_pred'] - audio_df['valence_true']
    audio_df['arousal_error'] = audio_df['arousal_pred'] - audio_df['arousal_true']
    
    # Plot 1: Valence error distribution
    axes[0, 0].hist(audio_df['valence_error'], bins=30, alpha=0.7, color='#FF6B6B', edgecolor='black')
    axes[0, 0].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Valence Error Distribution\n(Audio Level)', fontweight='bold')
    axes[0, 0].set_xlabel('Prediction Error')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Arousal error distribution
    axes[0, 1].hist(audio_df['arousal_error'], bins=30, alpha=0.7, color='#4ECDC4', edgecolor='black')
    axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Arousal Error Distribution\n(Audio Level)', fontweight='bold')
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: True vs Predicted distributions
    axes[0, 2].hist(audio_df['valence_true'], bins=20, alpha=0.5, label='True Valence', color='#FF6B6B')
    axes[0, 2].hist(audio_df['valence_pred'], bins=20, alpha=0.5, label='Pred Valence', color='#FF4444')
    axes[0, 2].hist(audio_df['arousal_true'], bins=20, alpha=0.5, label='True Arousal', color='#4ECDC4')
    axes[0, 2].hist(audio_df['arousal_pred'], bins=20, alpha=0.5, label='Pred Arousal', color='#2E8B8B')
    axes[0, 2].set_title('Value Distributions\n(Audio Level)', fontweight='bold')
    axes[0, 2].set_xlabel('Emotion Value')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Performance by emotion range (segment level)
    segment_df['valence_range'] = pd.cut(segment_df['valence_true'], bins=5, labels=['Very Low', 'Low', 'Mid', 'High', 'Very High'])
    segment_df['arousal_range'] = pd.cut(segment_df['arousal_true'], bins=5, labels=['Very Low', 'Low', 'Mid', 'High', 'Very High'])
    
    # Plot 4: Performance by valence range
    valence_mae_by_range = segment_df.groupby('valence_range').apply(
        lambda x: np.mean(np.abs(x['valence_pred'] - x['valence_true']))
    )
    axes[1, 0].bar(range(len(valence_mae_by_range)), valence_mae_by_range, color='#FF6B6B', alpha=0.7)
    axes[1, 0].set_title('MAE by Valence Range\n(Segment Level)', fontweight='bold')
    axes[1, 0].set_xlabel('Valence Range')
    axes[1, 0].set_ylabel('Mean Absolute Error')
    axes[1, 0].set_xticks(range(len(valence_mae_by_range)))
    axes[1, 0].set_xticklabels(valence_mae_by_range.index, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Performance by arousal range
    arousal_mae_by_range = segment_df.groupby('arousal_range').apply(
        lambda x: np.mean(np.abs(x['arousal_pred'] - x['arousal_true']))
    )
    axes[1, 1].bar(range(len(arousal_mae_by_range)), arousal_mae_by_range, color='#4ECDC4', alpha=0.7)
    axes[1, 1].set_title('MAE by Arousal Range\n(Segment Level)', fontweight='bold')
    axes[1, 1].set_xlabel('Arousal Range')
    axes[1, 1].set_ylabel('Mean Absolute Error')
    axes[1, 1].set_xticks(range(len(arousal_mae_by_range)))
    axes[1, 1].set_xticklabels(arousal_mae_by_range.index, rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Correlation matrix
    corr_data = audio_df[['valence_true', 'valence_pred', 'arousal_true', 'arousal_pred']]
    correlation_matrix = corr_data.corr()
    im = axes[1, 2].imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    axes[1, 2].set_title('Correlation Matrix\n(Audio Level)', fontweight='bold')
    axes[1, 2].set_xticks(range(len(correlation_matrix.columns)))
    axes[1, 2].set_yticks(range(len(correlation_matrix.index)))
    axes[1, 2].set_xticklabels(correlation_matrix.columns, rotation=45)
    axes[1, 2].set_yticklabels(correlation_matrix.index)
    
    # Add correlation values as text
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            axes[1, 2].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                           ha='center', va='center', fontweight='bold')
    
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'summary_statistics.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Summary statistics plots saved")


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description='Generate emotion prediction visualizations')
    parser.add_argument('prediction_dir', type=str, help='Directory containing prediction CSV files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.prediction_dir):
        print(f"Error: Directory {args.prediction_dir} does not exist")
        return
    
    create_emotion_visualizations(args.prediction_dir)


if __name__ == '__main__':
    main() 