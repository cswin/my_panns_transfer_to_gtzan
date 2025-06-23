#!/usr/bin/env python3
"""
Emotion prediction visualization module.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_emotion_visualizations(output_dir):
    """Create emotion prediction visualizations."""
    print(f"📊 Creating emotion visualizations from: {output_dir}")
    
    # Check if required files exist
    segment_csv = os.path.join(output_dir, 'segment_predictions.csv')
    audio_csv = os.path.join(output_dir, 'audio_predictions.csv')
    
    if not os.path.exists(segment_csv):
        print(f"❌ Segment predictions file not found: {segment_csv}")
        return
    
    if not os.path.exists(audio_csv):
        print(f"❌ Audio predictions file not found: {audio_csv}")
        return
    
    # Load data
    print("📖 Loading prediction data...")
    segment_df = pd.read_csv(segment_csv)
    audio_df = pd.read_csv(audio_csv)
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate basic scatter plots
    create_scatter_plots(audio_df, segment_df, plots_dir)
    
    print(f"✅ Visualizations saved in: {plots_dir}")

def create_scatter_plots(audio_df, segment_df, plots_dir):
    """Create scatter plots with metrics, fit line, and perfect prediction line."""
    print("  📈 Creating scatter plots...")
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_settings = [
        {
            'true': 'valence_true',
            'pred': 'valence_pred',
            'color': 'blue',
            'title': 'Audio-Level Valence Predictions',
            'xlabel': 'True Valence',
            'ylabel': 'Predicted Valence',
            'ax': axes[0]
        },
        {
            'true': 'arousal_true',
            'pred': 'arousal_pred',
            'color': 'red',
            'title': 'Audio-Level Arousal Predictions',
            'xlabel': 'True Arousal',
            'ylabel': 'Predicted Arousal',
            'ax': axes[1]
        }
    ]

    for setting in plot_settings:
        y_true = audio_df[setting['true']]
        y_pred = audio_df[setting['pred']]
        ax = setting['ax']
        # Scatter
        ax.scatter(y_true, y_pred, alpha=0.4, s=50, color=setting['color'])
        # Perfect prediction line
        ax.plot([-1, 1], [-1, 1], color='gray', lw=1, label='Perfect prediction')
        # Fit line
        fit = np.polyfit(y_true, y_pred, 1)
        fit_fn = np.poly1d(fit)
        x_vals = np.linspace(-1, 1, 100)
        ax.plot(x_vals, fit_fn(x_vals), 'r--', lw=2, label='Fit line')
        # Metrics
        r, p = pearsonr(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        metrics_text = (
            f"r = {r:.3f}\n"
            f"R² = {r2:.3f}\n"
            f"RMSE = {rmse:.3f}\n"
            f"MAE = {mae:.3f}\n"
            f"p = {p:.1e}"
        )
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        # Labels and title
        ax.set_xlabel(setting['xlabel'], fontsize=14, fontweight='bold')
        ax.set_ylabel(setting['ylabel'], fontsize=14, fontweight='bold')
        ax.set_title(setting['title'], fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        # Legend
        ax.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'audio_scatter_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ Saved: audio_scatter_plots.png")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python emotion_visualize.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    if not os.path.exists(output_dir):
        print(f"Error: Output directory does not exist: {output_dir}")
        sys.exit(1)
    
    create_emotion_visualizations(output_dir)
