#!/usr/bin/env python3

import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_training_curves(log_path):
    """Parse a training log file and extract training curves."""
    if not os.path.exists(log_path):
        return None
    
    curves = {
        'iterations': [],
        'val_audio_mae': [],
        'val_audio_pearson': [],
        'val_valence_mae': [],
        'val_arousal_mae': [],
        'val_valence_pearson': [],
        'val_arousal_pearson': [],
        'val_segment_mae': [],
        'val_segment_pearson': [],
        'train_loss': [],
        'file_path': log_path
    }
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        current_iteration = 0
        
        for line in lines:
            line = line.strip()
            
            # Look for iteration information
            if 'Iteration:' in line:
                iteration_match = re.search(r'Iteration:\s*(\d+)', line)
                if iteration_match:
                    current_iteration = int(iteration_match.group(1))
            
            # Look for training loss (appears before validation metrics)
            if 'Train loss:' in line and current_iteration > 0:
                loss_match = re.search(r'Train loss:\s*([\d.]+)', line)
                if loss_match:
                    curves['train_loss'].append(float(loss_match.group(1)))
            
            # Look for validation metrics
            if 'Validate Audio Mean MAE:' in line and current_iteration > 0:
                mae_match = re.search(r'Validate Audio Mean MAE:\s*([\d.]+)', line)
                if mae_match:
                    curves['iterations'].append(current_iteration)
                    curves['val_audio_mae'].append(float(mae_match.group(1)))
            
            if 'Validate Audio Mean Pearson:' in line:
                pearson_match = re.search(r'Validate Audio Mean Pearson:\s*([\d.]+)', line)
                if pearson_match:
                    curves['val_audio_pearson'].append(float(pearson_match.group(1)))
            
            # Parse valence and arousal MAE
            if 'Validate Audio Valence MAE:' in line and 'Arousal MAE:' in line:
                valence_match = re.search(r'Valence MAE:\s*([\d.]+)', line)
                arousal_match = re.search(r'Arousal MAE:\s*([\d.]+)', line)
                if valence_match and arousal_match:
                    curves['val_valence_mae'].append(float(valence_match.group(1)))
                    curves['val_arousal_mae'].append(float(arousal_match.group(1)))
            
            # Parse valence and arousal Pearson correlations
            if 'Validate Audio Valence Pearson:' in line and 'Arousal Pearson:' in line:
                valence_pearson_match = re.search(r'Valence Pearson:\s*([\d.]+)', line)
                arousal_pearson_match = re.search(r'Arousal Pearson:\s*([\d.]+)', line)
                if valence_pearson_match and arousal_pearson_match and current_iteration > 0:
                    # Only add if we have a valid current iteration
                    if len(curves['val_valence_pearson']) < len(curves['iterations']):
                        curves['val_valence_pearson'].append(float(valence_pearson_match.group(1)))
                        curves['val_arousal_pearson'].append(float(arousal_pearson_match.group(1)))
            
            if 'Validate Segment Mean MAE:' in line:
                seg_mae_match = re.search(r'Validate Segment Mean MAE:\s*([\d.]+)', line)
                if seg_mae_match:
                    curves['val_segment_mae'].append(float(seg_mae_match.group(1)))
            
            if 'Validate Segment Mean Pearson:' in line:
                seg_pearson_match = re.search(r'Validate Segment Mean Pearson:\s*([\d.]+)', line)
                if seg_pearson_match:
                    curves['val_segment_pearson'].append(float(seg_pearson_match.group(1)))
    
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None
    
    return curves

def find_latest_log_for_model(workspace_path, model_name):
    """Find the most recent log file for a specific model."""
    log_pattern = os.path.join(workspace_path, "logs", "**", f"*{model_name}*", "**", "*.log")
    log_files = glob.glob(log_pattern, recursive=True)
    
    if not log_files:
        return None
    
    # Sort by modification time and get the most recent
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return log_files[0]

def plot_single_metric(baseline_curves, lrm_curves, metric_key, title, ylabel, filename, higher_better=False):
    """Plot a single training metric comparison."""
    plt.figure(figsize=(12, 8))
    
    if baseline_curves['iterations'] and baseline_curves[metric_key]:
        plt.plot(baseline_curves['iterations'], baseline_curves[metric_key], 
                'b-', linewidth=3, label='Baseline (NewAffective)', marker='o', markersize=5, alpha=0.8)
    
    if lrm_curves['iterations'] and lrm_curves[metric_key]:
        plt.plot(lrm_curves['iterations'], lrm_curves[metric_key], 
                'r-', linewidth=3, label='LRM (Feedback)', marker='s', markersize=5, alpha=0.8)
    
    plt.xlabel('Training Iteration', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add performance indicator
    if higher_better:
        plt.text(0.02, 0.02, '(Higher is Better)', transform=plt.gca().transAxes, 
                fontsize=10, style='italic', alpha=0.7)
    else:
        plt.text(0.02, 0.98, '(Lower is Better)', transform=plt.gca().transAxes, 
                fontsize=10, style='italic', alpha=0.7, verticalalignment='top')
    
    # Save the plot
    os.makedirs('training_curves', exist_ok=True)
    output_file = f'training_curves/{filename}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 {title} saved to: {output_file}")
    
    # Also save as PDF
    output_pdf = f'training_curves/{filename}.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    
    plt.show()
    
    # Print summary
    if baseline_curves[metric_key]:
        if higher_better:
            best_baseline = max(baseline_curves[metric_key])
        else:
            best_baseline = min(baseline_curves[metric_key])
        final_baseline = baseline_curves[metric_key][-1]
        print(f"Baseline: {best_baseline:.4f} (best) → {final_baseline:.4f} (final)")
    
    if lrm_curves[metric_key]:
        if higher_better:
            best_lrm = max(lrm_curves[metric_key])
        else:
            best_lrm = min(lrm_curves[metric_key])
        final_lrm = lrm_curves[metric_key][-1]
        print(f"LRM: {best_lrm:.4f} (best) → {final_lrm:.4f} (final)")

def plot_training_curves():
    """Plot training curves for both baseline and LRM models."""
    
    # Find log files for both models
    baseline_log = find_latest_log_for_model("workspaces/emotion_regression", "FeatureEmotionRegression_Cnn6_NewAffective")
    lrm_log = find_latest_log_for_model("workspaces/emotion_feedback", "FeatureEmotionRegression_Cnn6_LRM")
    
    if not baseline_log:
        print("❌ Could not find baseline model log")
        return
    
    if not lrm_log:
        print("❌ Could not find LRM model log")
        return
    
    print(f"📊 Parsing baseline log: {os.path.basename(baseline_log)}")
    baseline_curves = parse_training_curves(baseline_log)
    
    print(f"🔄 Parsing LRM log: {os.path.basename(lrm_log)}")
    lrm_curves = parse_training_curves(lrm_log)
    
    if not baseline_curves or not lrm_curves:
        print("❌ Could not parse training curves")
        return
    
    print(f"\n📈 GENERATING AUDIO-LEVEL PERFORMANCE PLOTS:")
    print("=" * 60)
    
    # Plot 1: Valence Pearson Correlation
    if baseline_curves['val_valence_pearson'] or lrm_curves['val_valence_pearson']:
        print(f"\n1️⃣ Valence Pearson Correlation:")
        plot_single_metric(
            baseline_curves, lrm_curves, 
            'val_valence_pearson',
            'Audio-Level Valence Performance (Pearson Correlation)',
            'Valence Pearson Correlation',
            'valence_pearson_comparison',
            higher_better=True
        )
    else:
        print("⚠️ No valence Pearson data found in logs")
    
    # Plot 2: Arousal Pearson Correlation  
    if baseline_curves['val_arousal_pearson'] or lrm_curves['val_arousal_pearson']:
        print(f"\n2️⃣ Arousal Pearson Correlation:")
        plot_single_metric(
            baseline_curves, lrm_curves,
            'val_arousal_pearson', 
            'Audio-Level Arousal Performance (Pearson Correlation)',
            'Arousal Pearson Correlation',
            'arousal_pearson_comparison',
            higher_better=True
        )
    else:
        print("⚠️ No arousal Pearson data found in logs")
    
    # Plot 3: Valence MAE
    if baseline_curves['val_valence_mae'] or lrm_curves['val_valence_mae']:
        print(f"\n3️⃣ Valence MAE:")
        plot_single_metric(
            baseline_curves, lrm_curves,
            'val_valence_mae',
            'Audio-Level Valence Performance (Mean Absolute Error)',
            'Valence MAE',
            'valence_mae_comparison',
            higher_better=False
        )
    else:
        print("⚠️ No valence MAE data found in logs")
    
    # Plot 4: Arousal MAE
    if baseline_curves['val_arousal_mae'] or lrm_curves['val_arousal_mae']:
        print(f"\n4️⃣ Arousal MAE:")
        plot_single_metric(
            baseline_curves, lrm_curves,
            'val_arousal_mae',
            'Audio-Level Arousal Performance (Mean Absolute Error)', 
            'Arousal MAE',
            'arousal_mae_comparison',
            higher_better=False
        )
    else:
        print("⚠️ No arousal MAE data found in logs")

def main():
    """Main function to generate training curve plots."""
    print("📊 GENERATING AUDIO-LEVEL VALENCE & AROUSAL PLOTS")
    print("=" * 50)
    
    plot_training_curves()
    
    print(f"\n✅ Audio-level performance analysis completed!")
    print(f"💡 Generated 4 separate plots focusing on:")
    print(f"   - Valence Pearson Correlation (higher is better)")
    print(f"   - Arousal Pearson Correlation (higher is better)")
    print(f"   - Valence MAE (lower is better)")
    print(f"   - Arousal MAE (lower is better)")
    print(f"\n🔍 Note: The training script now logs separate valence/arousal")
    print(f"   Pearson correlations. You may need to retrain to get this data")

if __name__ == '__main__':
    main() 