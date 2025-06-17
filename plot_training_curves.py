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
            
            if 'Validate Audio Valence MAE:' in line and 'Arousal MAE:' in line:
                valence_match = re.search(r'Valence MAE:\s*([\d.]+)', line)
                arousal_match = re.search(r'Arousal MAE:\s*([\d.]+)', line)
                if valence_match and arousal_match:
                    curves['val_valence_mae'].append(float(valence_match.group(1)))
                    curves['val_arousal_mae'].append(float(arousal_match.group(1)))
            
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
    
    # Create the plots
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Curves: Baseline vs LRM Emotion Regression', fontsize=16, fontweight='bold')
    
    # Plot 1: Audio MAE over iterations
    ax1 = axes[0, 0]
    if baseline_curves['iterations'] and baseline_curves['val_audio_mae']:
        ax1.plot(baseline_curves['iterations'], baseline_curves['val_audio_mae'], 
                'b-', linewidth=2, label='Baseline (NewAffective)', marker='o', markersize=3)
    if lrm_curves['iterations'] and lrm_curves['val_audio_mae']:
        ax1.plot(lrm_curves['iterations'], lrm_curves['val_audio_mae'], 
                'r-', linewidth=2, label='LRM (Feedback)', marker='s', markersize=3)
    
    ax1.set_xlabel('Training Iteration')
    ax1.set_ylabel('Validation Audio MAE')
    ax1.set_title('Audio-Level Mean Absolute Error\n(Lower is Better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Audio Pearson correlation over iterations
    ax2 = axes[0, 1]
    if baseline_curves['iterations'] and baseline_curves['val_audio_pearson']:
        ax2.plot(baseline_curves['iterations'], baseline_curves['val_audio_pearson'], 
                'b-', linewidth=2, label='Baseline (NewAffective)', marker='o', markersize=3)
    if lrm_curves['iterations'] and lrm_curves['val_audio_pearson']:
        ax2.plot(lrm_curves['iterations'], lrm_curves['val_audio_pearson'], 
                'r-', linewidth=2, label='LRM (Feedback)', marker='s', markersize=3)
    
    ax2.set_xlabel('Training Iteration')
    ax2.set_ylabel('Validation Audio Pearson Correlation')
    ax2.set_title('Audio-Level Pearson Correlation\n(Higher is Better)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Valence vs Arousal MAE
    ax3 = axes[1, 0]
    if baseline_curves['iterations'] and baseline_curves['val_valence_mae']:
        ax3.plot(baseline_curves['iterations'], baseline_curves['val_valence_mae'], 
                'b-', linewidth=2, label='Baseline Valence', marker='o', markersize=3)
        ax3.plot(baseline_curves['iterations'], baseline_curves['val_arousal_mae'], 
                'b--', linewidth=2, label='Baseline Arousal', marker='o', markersize=3)
    if lrm_curves['iterations'] and lrm_curves['val_valence_mae']:
        ax3.plot(lrm_curves['iterations'], lrm_curves['val_valence_mae'], 
                'r-', linewidth=2, label='LRM Valence', marker='s', markersize=3)
        ax3.plot(lrm_curves['iterations'], lrm_curves['val_arousal_mae'], 
                'r--', linewidth=2, label='LRM Arousal', marker='s', markersize=3)
    
    ax3.set_xlabel('Training Iteration')
    ax3.set_ylabel('Validation MAE')
    ax3.set_title('Valence vs Arousal MAE\n(Lower is Better)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Segment-level performance
    ax4 = axes[1, 1]
    if baseline_curves['iterations'] and baseline_curves['val_segment_mae']:
        ax4.plot(baseline_curves['iterations'], baseline_curves['val_segment_mae'], 
                'b-', linewidth=2, label='Baseline Segment MAE', marker='o', markersize=3)
    if baseline_curves['iterations'] and baseline_curves['val_segment_pearson']:
        # Scale Pearson to same range as MAE for visualization
        scaled_pearson = [p * 0.8 for p in baseline_curves['val_segment_pearson']]  # Scale down for visibility
        ax4.plot(baseline_curves['iterations'], scaled_pearson, 
                'b:', linewidth=2, label='Baseline Segment Pearson (×0.8)', marker='o', markersize=3)
    
    if lrm_curves['iterations'] and lrm_curves['val_segment_mae']:
        ax4.plot(lrm_curves['iterations'], lrm_curves['val_segment_mae'], 
                'r-', linewidth=2, label='LRM Segment MAE', marker='s', markersize=3)
    if lrm_curves['iterations'] and lrm_curves['val_segment_pearson']:
        # Scale Pearson to same range as MAE for visualization
        scaled_pearson = [p * 0.8 for p in lrm_curves['val_segment_pearson']]  # Scale down for visibility
        ax4.plot(lrm_curves['iterations'], scaled_pearson, 
                'r:', linewidth=2, label='LRM Segment Pearson (×0.8)', marker='s', markersize=3)
    
    ax4.set_xlabel('Training Iteration')
    ax4.set_ylabel('Validation Metrics')
    ax4.set_title('Segment-Level Performance\n(MAE: Lower Better, Pearson: Higher Better)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create output directory
    os.makedirs('training_curves', exist_ok=True)
    
    # Save the plot
    output_file = 'training_curves/baseline_vs_lrm_training_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to: {output_file}")
    
    # Also save as PDF for high quality
    output_pdf = 'training_curves/baseline_vs_lrm_training_curves.pdf'
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"📄 High-quality PDF saved to: {output_pdf}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print(f"\n📈 TRAINING CURVE SUMMARY:")
    print(f"Baseline Model:")
    if baseline_curves['val_audio_mae']:
        print(f"  - Audio MAE: {min(baseline_curves['val_audio_mae']):.4f} (best) → {baseline_curves['val_audio_mae'][-1]:.4f} (final)")
    if baseline_curves['val_audio_pearson']:
        print(f"  - Audio Pearson: {max(baseline_curves['val_audio_pearson']):.4f} (best) → {baseline_curves['val_audio_pearson'][-1]:.4f} (final)")
    
    print(f"\nLRM Model:")
    if lrm_curves['val_audio_mae']:
        print(f"  - Audio MAE: {min(lrm_curves['val_audio_mae']):.4f} (best) → {lrm_curves['val_audio_mae'][-1]:.4f} (final)")
    if lrm_curves['val_audio_pearson']:
        print(f"  - Audio Pearson: {max(lrm_curves['val_audio_pearson']):.4f} (best) → {lrm_curves['val_audio_pearson'][-1]:.4f} (final)")

def main():
    """Main function to generate training curve plots."""
    print("📊 GENERATING TRAINING CURVES")
    print("=" * 50)
    
    plot_training_curves()
    
    print(f"\n✅ Training curve analysis completed!")
    print(f"💡 The plots show validation accuracy over training iterations")
    print(f"   - Compare how both models learned over time")
    print(f"   - Look for convergence patterns and stability")
    print(f"   - Check if either model shows signs of overfitting")

if __name__ == '__main__':
    main() 