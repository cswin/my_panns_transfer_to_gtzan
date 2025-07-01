#!/usr/bin/env python3
"""
Visualize Steering Signal Effects

This script creates comprehensive visualizations of steering signal effects including:
1. Scatter plots comparing baseline vs steered predictions
2. Direction accuracy analysis
3. Category effectiveness comparison
4. Steering magnitude analysis

Usage:
    python scripts/visualize_steering_effects.py --results_file steering_test_results/steering_test_results.json --output_dir steering_plots
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_steering_results(results_file):
    """Load steering test results from JSON file."""
    print(f"Loading steering results from: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame for easier analysis
    data = []
    for result in results:
        audio_name = result.get('audio_name', 'unknown')
        target = result.get('target', {})
        baseline = result.get('baseline', {})
        selected_category = result.get('selected_category', None)
        
        # Base row
        row = {
            'audio_name': audio_name,
            'target_valence': target.get('valence', np.nan),
            'target_arousal': target.get('arousal', np.nan),
            'baseline_valence': baseline.get('valence', np.nan),
            'baseline_arousal': baseline.get('arousal', np.nan),
            'selected_category': selected_category
        }
        
        # Add steered results
        for key, value in result.items():
            if key not in ['audio_name', 'target', 'baseline', 'selected_category'] and isinstance(value, dict):
                if 'valence' in value and 'arousal' in value:
                    row[f'{key}_valence'] = value['valence']
                    row[f'{key}_arousal'] = value['arousal']
                    row[f'{key}_valence_change'] = value['valence'] - baseline.get('valence', 0)
                    row[f'{key}_arousal_change'] = value['arousal'] - baseline.get('arousal', 0)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples with steering results")
    return df

def create_baseline_vs_steered_scatter(df, output_dir):
    """Create scatter plots comparing baseline vs steered predictions."""
    print("Creating baseline vs steered scatter plots...")
    
    # Find steered columns (exclude baseline, target, and metadata)
    steered_cols = [col for col in df.columns if col.endswith('_valence') and not col.startswith(('target_', 'baseline_'))]
    steered_categories = [col.replace('_valence', '') for col in steered_cols]
    
    if not steered_categories:
        print("No steered predictions found in data")
        return
    
    # Create figure with subplots
    n_categories = len(steered_categories)
    fig, axes = plt.subplots(2, max(2, n_categories), figsize=(6*n_categories, 12))
    if n_categories == 1:
        axes = axes.reshape(2, 1)
    
    # Plot for each category
    for i, category in enumerate(steered_categories):
        val_col = f'{category}_valence'
        aro_col = f'{category}_arousal'
        
        if val_col not in df.columns or aro_col not in df.columns:
            continue
        
        # Filter valid data
        mask = ~(df[val_col].isna() | df[aro_col].isna() | df['baseline_valence'].isna() | df['baseline_arousal'].isna())
        if not mask.any():
            continue
        
        data_subset = df[mask]
        
        # Valence plot
        ax_val = axes[0, min(i, axes.shape[1]-1)]
        baseline_val = data_subset['baseline_valence']
        steered_val = data_subset[val_col]
        target_val = data_subset['target_valence']
        
        # Scatter plot
        ax_val.scatter(baseline_val, steered_val, alpha=0.6, s=50, c='blue', label='Steered vs Baseline')
        ax_val.scatter(baseline_val, target_val, alpha=0.6, s=30, c='red', marker='x', label='Target')
        
        # Perfect prediction line
        ax_val.plot([-1, 1], [-1, 1], 'gray', linestyle='--', alpha=0.7, label='No change')
        
        # Fit line
        if len(baseline_val) > 1:
            fit = np.polyfit(baseline_val, steered_val, 1)
            fit_fn = np.poly1d(fit)
            x_vals = np.linspace(baseline_val.min(), baseline_val.max(), 100)
            ax_val.plot(x_vals, fit_fn(x_vals), 'green', linestyle='-', linewidth=2, label='Fit line')
        
        # Calculate metrics (only if we have enough data points)
        if len(baseline_val) >= 2:
            r_val, p_val = pearsonr(baseline_val, steered_val)
            rmse_val = np.sqrt(mean_squared_error(baseline_val, steered_val))
            mae_val = mean_absolute_error(baseline_val, steered_val)
        else:
            r_val, p_val = np.nan, np.nan
            rmse_val = np.nan
            mae_val = np.nan
        
        # Direction accuracy
        val_changes = steered_val - baseline_val
        target_changes = target_val - baseline_val
        val_correct = np.sum((val_changes > 0) == (target_changes > 0)) / len(val_changes) * 100
        
        r_text = f"{r_val:.3f}" if not np.isnan(r_val) else "N/A"
        rmse_text = f"{rmse_val:.3f}" if not np.isnan(rmse_val) else "N/A"
        mae_text = f"{mae_val:.3f}" if not np.isnan(mae_val) else "N/A"
        
        metrics_text = (
            f"r = {r_text}\n"
            f"RMSE = {rmse_text}\n"
            f"MAE = {mae_text}\n"
            f"Dir Acc = {val_correct:.1f}%"
        )
        
        ax_val.text(0.05, 0.95, metrics_text, transform=ax_val.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_val.set_xlabel('Baseline Valence', fontsize=12)
        ax_val.set_ylabel('Steered Valence', fontsize=12)
        ax_val.set_title(f'{category.replace("_", " ").title()} - Valence', fontsize=14, fontweight='bold')
        ax_val.grid(True, alpha=0.3)
        ax_val.legend(fontsize=8)
        ax_val.set_xlim([-1, 1])
        ax_val.set_ylim([-1, 1])
        
        # Arousal plot
        ax_aro = axes[1, min(i, axes.shape[1]-1)]
        baseline_aro = data_subset['baseline_arousal']
        steered_aro = data_subset[aro_col]
        target_aro = data_subset['target_arousal']
        
        # Scatter plot
        ax_aro.scatter(baseline_aro, steered_aro, alpha=0.6, s=50, c='orange', label='Steered vs Baseline')
        ax_aro.scatter(baseline_aro, target_aro, alpha=0.6, s=30, c='red', marker='x', label='Target')
        
        # Perfect prediction line
        ax_aro.plot([-1, 1], [-1, 1], 'gray', linestyle='--', alpha=0.7, label='No change')
        
        # Fit line
        if len(baseline_aro) > 1:
            fit = np.polyfit(baseline_aro, steered_aro, 1)
            fit_fn = np.poly1d(fit)
            x_vals = np.linspace(baseline_aro.min(), baseline_aro.max(), 100)
            ax_aro.plot(x_vals, fit_fn(x_vals), 'green', linestyle='-', linewidth=2, label='Fit line')
        
        # Calculate metrics (only if we have enough data points)
        if len(baseline_aro) >= 2:
            r_aro, p_aro = pearsonr(baseline_aro, steered_aro)
            rmse_aro = np.sqrt(mean_squared_error(baseline_aro, steered_aro))
            mae_aro = mean_absolute_error(baseline_aro, steered_aro)
        else:
            r_aro, p_aro = np.nan, np.nan
            rmse_aro = np.nan
            mae_aro = np.nan
        
        # Direction accuracy
        aro_changes = steered_aro - baseline_aro
        target_changes = target_aro - baseline_aro
        aro_correct = np.sum((aro_changes > 0) == (target_changes > 0)) / len(aro_changes) * 100
        
        r_aro_text = f"{r_aro:.3f}" if not np.isnan(r_aro) else "N/A"
        rmse_aro_text = f"{rmse_aro:.3f}" if not np.isnan(rmse_aro) else "N/A"
        mae_aro_text = f"{mae_aro:.3f}" if not np.isnan(mae_aro) else "N/A"
        
        metrics_text = (
            f"r = {r_aro_text}\n"
            f"RMSE = {rmse_aro_text}\n"
            f"MAE = {mae_aro_text}\n"
            f"Dir Acc = {aro_correct:.1f}%"
        )
        
        ax_aro.text(0.05, 0.95, metrics_text, transform=ax_aro.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax_aro.set_xlabel('Baseline Arousal', fontsize=12)
        ax_aro.set_ylabel('Steered Arousal', fontsize=12)
        ax_aro.set_title(f'{category.replace("_", " ").title()} - Arousal', fontsize=14, fontweight='bold')
        ax_aro.grid(True, alpha=0.3)
        ax_aro.legend(fontsize=8)
        ax_aro.set_xlim([-1, 1])
        ax_aro.set_ylim([-1, 1])
    
    # Hide unused subplots
    for i in range(n_categories, axes.shape[1]):
        axes[0, i].set_visible(False)
        axes[1, i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_vs_steered_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: baseline_vs_steered_scatter.png")

def create_target_vs_steered_scatter(df, output_dir):
    """Create scatter plots comparing target vs steered predictions (like existing emotion plots)."""
    print("Creating target vs steered scatter plots...")
    
    # Find steered columns
    steered_cols = [col for col in df.columns if col.endswith('_valence') and not col.startswith(('target_', 'baseline_'))]
    steered_categories = [col.replace('_valence', '') for col in steered_cols]
    
    if not steered_categories:
        print("No steered predictions found")
        return
    
    # Create combined plot similar to existing audio_scatter_plots.png
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(steered_categories)))
    
    # Valence plot
    ax = axes[0]
    for i, category in enumerate(steered_categories):
        val_col = f'{category}_valence'
        
        if val_col not in df.columns:
            continue
        
        # Filter valid data
        mask = ~(df[val_col].isna() | df['target_valence'].isna())
        if not mask.any():
            continue
        
        data_subset = df[mask]
        target_val = data_subset['target_valence']
        steered_val = data_subset[val_col]
        
        # Scatter plot
        ax.scatter(target_val, steered_val, alpha=0.6, s=50, c=[colors[i]], 
                  label=category.replace('_', ' ').title())
    
    # Perfect prediction line
    ax.plot([-1, 1], [-1, 1], 'gray', linestyle='--', alpha=0.7, label='Perfect prediction')
    
    # Calculate overall metrics (using all data)
    all_targets = []
    all_steered = []
    for category in steered_categories:
        val_col = f'{category}_valence'
        if val_col in df.columns:
            mask = ~(df[val_col].isna() | df['target_valence'].isna())
            if mask.any():
                all_targets.extend(df[mask]['target_valence'].tolist())
                all_steered.extend(df[mask][val_col].tolist())
    
    if all_targets and all_steered:
        # Fit line
        fit = np.polyfit(all_targets, all_steered, 1)
        fit_fn = np.poly1d(fit)
        x_vals = np.linspace(-1, 1, 100)
        ax.plot(x_vals, fit_fn(x_vals), 'red', linestyle='--', linewidth=2, label='Fit line')
        
        # Metrics
        r, p = pearsonr(all_targets, all_steered)
        r2 = r2_score(all_targets, all_steered)
        rmse = np.sqrt(mean_squared_error(all_targets, all_steered))
        mae = mean_absolute_error(all_targets, all_steered)
        
        metrics_text = (
            f"r = {r:.3f}\n"
            f"R² = {r2:.3f}\n"
            f"RMSE = {rmse:.3f}\n"
            f"MAE = {mae:.3f}\n"
            f"p = {p:.1e}"
        )
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Target Valence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Steered Valence', fontsize=14, fontweight='bold')
    ax.set_title('Audio-Level Valence Steering Effects', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    
    # Arousal plot
    ax = axes[1]
    for i, category in enumerate(steered_categories):
        aro_col = f'{category}_arousal'
        
        if aro_col not in df.columns:
            continue
        
        # Filter valid data
        mask = ~(df[aro_col].isna() | df['target_arousal'].isna())
        if not mask.any():
            continue
        
        data_subset = df[mask]
        target_aro = data_subset['target_arousal']
        steered_aro = data_subset[aro_col]
        
        # Scatter plot
        ax.scatter(target_aro, steered_aro, alpha=0.6, s=50, c=[colors[i]], 
                  label=category.replace('_', ' ').title())
    
    # Perfect prediction line
    ax.plot([-1, 1], [-1, 1], 'gray', linestyle='--', alpha=0.7, label='Perfect prediction')
    
    # Calculate overall metrics (using all data)
    all_targets = []
    all_steered = []
    for category in steered_categories:
        aro_col = f'{category}_arousal'
        if aro_col in df.columns:
            mask = ~(df[aro_col].isna() | df['target_arousal'].isna())
            if mask.any():
                all_targets.extend(df[mask]['target_arousal'].tolist())
                all_steered.extend(df[mask][aro_col].tolist())
    
    if all_targets and all_steered:
        # Fit line
        fit = np.polyfit(all_targets, all_steered, 1)
        fit_fn = np.poly1d(fit)
        x_vals = np.linspace(-1, 1, 100)
        ax.plot(x_vals, fit_fn(x_vals), 'red', linestyle='--', linewidth=2, label='Fit line')
        
        # Metrics
        r, p = pearsonr(all_targets, all_steered)
        r2 = r2_score(all_targets, all_steered)
        rmse = np.sqrt(mean_squared_error(all_targets, all_steered))
        mae = mean_absolute_error(all_targets, all_steered)
        
        metrics_text = (
            f"r = {r:.3f}\n"
            f"R² = {r2:.3f}\n"
            f"RMSE = {rmse:.3f}\n"
            f"MAE = {mae:.3f}\n"
            f"p = {p:.1e}"
        )
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Target Arousal', fontsize=14, fontweight='bold')
    ax.set_ylabel('Steered Arousal', fontsize=14, fontweight='bold')
    ax.set_title('Audio-Level Arousal Steering Effects', fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'audio_steering_scatter_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: audio_steering_scatter_plots.png")

def create_summary_report(df, output_dir):
    """Create a summary report of steering effects."""
    print("Creating summary report...")
    
    # Find steered columns
    steered_cols = [col for col in df.columns if col.endswith('_valence') and not col.startswith(('target_', 'baseline_'))]
    steered_categories = [col.replace('_valence', '') for col in steered_cols]
    
    report_lines = []
    report_lines.append("# Steering Signal Effects Summary Report")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    report_lines.append(f"**Total Samples:** {len(df)}")
    report_lines.append(f"**Steering Categories:** {len(steered_categories)}")
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("## Overall Statistics")
    report_lines.append("")
    
    for category in steered_categories:
        val_col = f'{category}_valence'
        aro_col = f'{category}_arousal'
        val_change_col = f'{category}_valence_change'
        aro_change_col = f'{category}_arousal_change'
        
        if val_col not in df.columns or aro_col not in df.columns:
            continue
        
        # Filter valid data
        mask = ~(df[val_col].isna() | df[aro_col].isna() | 
                df['baseline_valence'].isna() | df['baseline_arousal'].isna() |
                df['target_valence'].isna() | df['target_arousal'].isna())
        
        if not mask.any():
            continue
        
        data_subset = df[mask]
        
        # Calculate metrics
        val_changes = data_subset[val_change_col] if val_change_col in df.columns else (data_subset[val_col] - data_subset['baseline_valence'])
        aro_changes = data_subset[aro_change_col] if aro_change_col in df.columns else (data_subset[aro_col] - data_subset['baseline_arousal'])
        
        target_val_changes = data_subset['target_valence'] - data_subset['baseline_valence']
        target_aro_changes = data_subset['target_arousal'] - data_subset['baseline_arousal']
        
        # Direction accuracy
        val_correct = np.sum((val_changes > 0) == (target_val_changes > 0)) / len(val_changes) * 100
        aro_correct = np.sum((aro_changes > 0) == (target_aro_changes > 0)) / len(aro_changes) * 100
        both_correct = np.sum(((val_changes > 0) == (target_val_changes > 0)) & 
                             ((aro_changes > 0) == (target_aro_changes > 0))) / len(val_changes) * 100
        
        report_lines.append(f"### {category.replace('_', ' ').title()}")
        report_lines.append(f"- **Samples:** {len(data_subset)}")
        report_lines.append(f"- **Mean Valence Change:** {np.mean(val_changes):.4f} ± {np.std(val_changes):.4f}")
        report_lines.append(f"- **Mean Arousal Change:** {np.mean(aro_changes):.4f} ± {np.std(aro_changes):.4f}")
        report_lines.append(f"- **Valence Direction Accuracy:** {val_correct:.1f}%")
        report_lines.append(f"- **Arousal Direction Accuracy:** {aro_correct:.1f}%")
        report_lines.append(f"- **Both Directions Correct:** {both_correct:.1f}%")
        report_lines.append("")
    
    # Save report
    report_path = os.path.join(output_dir, 'steering_effects_summary.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"  ✅ Saved: steering_effects_summary.md")

def main():
    parser = argparse.ArgumentParser(description='Visualize steering signal effects')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to steering test results JSON file')
    parser.add_argument('--output_dir', type=str, default='steering_plots',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    df = load_steering_results(args.results_file)
    
    if df.empty:
        print("No data to visualize")
        return
    
    # Create visualizations
    print("\n📊 Creating steering effect visualizations...")
    print("=" * 50)
    
    create_target_vs_steered_scatter(df, args.output_dir)
    create_baseline_vs_steered_scatter(df, args.output_dir)
    create_summary_report(df, args.output_dir)
    
    print(f"\n✅ All visualizations saved to: {args.output_dir}")
    print("Generated files:")
    print("  - audio_steering_scatter_plots.png (main steering effects plot)")
    print("  - baseline_vs_steered_scatter.png (detailed category plots)")
    print("  - steering_effects_summary.md (summary report)")

if __name__ == '__main__':
    main() 