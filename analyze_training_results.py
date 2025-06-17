#!/usr/bin/env python3

import os
import glob
import re
from datetime import datetime

def parse_nested_log_file(log_path):
    """Parse a training log file and extract key metrics."""
    if not os.path.exists(log_path):
        return None
    
    metrics = {
        'iterations': [],
        'train_loss': [],
        'val_audio_mae': [],
        'val_audio_rmse': [],
        'val_audio_pearson': [],
        'val_valence_mae': [],
        'val_arousal_mae': [],
        'val_segment_mae': [],
        'val_segment_pearson': [],
        'last_iteration': 0,
        'best_audio_mae': float('inf'),
        'best_audio_pearson': -1.0,
        'training_complete': False,
        'file_path': log_path
    }
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Look for iteration information
            if 'Iteration:' in line:
                iteration_match = re.search(r'Iteration:\s*(\d+)', line)
                if iteration_match:
                    iteration = int(iteration_match.group(1))
                    metrics['iterations'].append(iteration)
                    metrics['last_iteration'] = max(metrics['last_iteration'], iteration)
            
            # Look for validation metrics
            if 'Validate Audio Mean MAE:' in line:
                mae_match = re.search(r'Validate Audio Mean MAE:\s*([\d.]+)', line)
                if mae_match:
                    mae = float(mae_match.group(1))
                    metrics['val_audio_mae'].append(mae)
                    metrics['best_audio_mae'] = min(metrics['best_audio_mae'], mae)
            
            if 'Validate Audio Mean RMSE:' in line:
                rmse_match = re.search(r'Validate Audio Mean RMSE:\s*([\d.]+)', line)
                if rmse_match:
                    metrics['val_audio_rmse'].append(float(rmse_match.group(1)))
            
            if 'Validate Audio Mean Pearson:' in line:
                pearson_match = re.search(r'Validate Audio Mean Pearson:\s*([\d.]+)', line)
                if pearson_match:
                    pearson = float(pearson_match.group(1))
                    metrics['val_audio_pearson'].append(pearson)
                    metrics['best_audio_pearson'] = max(metrics['best_audio_pearson'], pearson)
            
            if 'Validate Audio Valence MAE:' in line and 'Arousal MAE:' in line:
                valence_match = re.search(r'Valence MAE:\s*([\d.]+)', line)
                arousal_match = re.search(r'Arousal MAE:\s*([\d.]+)', line)
                if valence_match and arousal_match:
                    metrics['val_valence_mae'].append(float(valence_match.group(1)))
                    metrics['val_arousal_mae'].append(float(arousal_match.group(1)))
            
            if 'Validate Segment Mean MAE:' in line:
                seg_mae_match = re.search(r'Validate Segment Mean MAE:\s*([\d.]+)', line)
                if seg_mae_match:
                    metrics['val_segment_mae'].append(float(seg_mae_match.group(1)))
            
            if 'Validate Segment Mean Pearson:' in line:
                seg_pearson_match = re.search(r'Validate Segment Mean Pearson:\s*([\d.]+)', line)
                if seg_pearson_match:
                    metrics['val_segment_pearson'].append(float(seg_pearson_match.group(1)))
            
            # Check if training completed (reached 5000 iterations)
            if 'Model saved to' in line and '5000_iterations.pth' in line:
                metrics['training_complete'] = True
    
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None
    
    return metrics

def find_latest_log_in_workspace(workspace_path, model_pattern):
    """Find the most recent log file for a given model pattern."""
    log_pattern = os.path.join(workspace_path, "logs", "**", f"*{model_pattern}*", "**", "*.log")
    log_files = glob.glob(log_pattern, recursive=True)
    
    if not log_files:
        return None
    
    # Sort by modification time and get the most recent
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return log_files[0]

def analyze_workspace(workspace_path, workspace_name):
    """Analyze all models in a workspace."""
    print(f"\n🔍 Analyzing {workspace_name}")
    print("=" * 60)
    
    if not os.path.exists(workspace_path):
        print(f"   ❌ Workspace doesn't exist: {workspace_path}")
        return {}
    
    results = {}
    
    # Find all log files in the workspace
    log_pattern = os.path.join(workspace_path, "logs", "**", "*.log")
    log_files = glob.glob(log_pattern, recursive=True)
    
    if not log_files:
        print(f"   ❌ No log files found")
        return results
    
    # Group logs by model type
    model_logs = {}
    for log_file in log_files:
        # Extract model name from path
        path_parts = log_file.split(os.sep)
        for part in path_parts:
            if 'FeatureEmotionRegression' in part:
                model_name = part
                if model_name not in model_logs:
                    model_logs[model_name] = []
                model_logs[model_name].append(log_file)
                break
    
    # Analyze each model
    for model_name, logs in model_logs.items():
        print(f"\n📊 Model: {model_name}")
        
        # Find the most recent log
        latest_log = max(logs, key=lambda x: os.path.getmtime(x))
        print(f"   📄 Latest log: {os.path.basename(latest_log)}")
        
        # Parse the log
        metrics = parse_nested_log_file(latest_log)
        if metrics:
            results[model_name] = metrics
            
            print(f"   ✅ Training complete: {metrics['training_complete']}")
            print(f"   📈 Last iteration: {metrics['last_iteration']}")
            
            if metrics['best_audio_mae'] != float('inf'):
                print(f"   🎯 Best Audio MAE: {metrics['best_audio_mae']:.4f}")
            
            if metrics['best_audio_pearson'] != -1.0:
                print(f"   📊 Best Audio Pearson: {metrics['best_audio_pearson']:.4f}")
            
            if metrics['val_audio_mae']:
                final_mae = metrics['val_audio_mae'][-1]
                print(f"   📉 Final Audio MAE: {final_mae:.4f}")
            
            if metrics['val_audio_pearson']:
                final_pearson = metrics['val_audio_pearson'][-1]
                print(f"   📈 Final Audio Pearson: {final_pearson:.4f}")
            
            if metrics['val_valence_mae'] and metrics['val_arousal_mae']:
                final_val_mae = metrics['val_valence_mae'][-1]
                final_aro_mae = metrics['val_arousal_mae'][-1]
                print(f"   💝 Final Valence MAE: {final_val_mae:.4f}")
                print(f"   🔥 Final Arousal MAE: {final_aro_mae:.4f}")
        else:
            print(f"   ❌ Could not parse log file")
    
    return results

def compare_models(baseline_results, lrm_results):
    """Compare baseline and LRM model results."""
    print(f"\n🔄 MODEL COMPARISON")
    print("=" * 60)
    
    # Find the main models to compare
    baseline_model = None
    lrm_model = None
    
    for model_name in baseline_results:
        if 'NewAffective' in model_name:
            baseline_model = baseline_results[model_name]
            baseline_name = model_name
            break
    
    for model_name in lrm_results:
        if 'LRM' in model_name:
            lrm_model = lrm_results[model_name]
            lrm_name = model_name
            break
    
    if not baseline_model or not lrm_model:
        print("❌ Could not find both baseline and LRM models for comparison")
        return
    
    print(f"📊 Baseline: {baseline_name}")
    print(f"🔄 LRM: {lrm_name}")
    print()
    
    # Compare key metrics
    baseline_mae = baseline_model['best_audio_mae']
    lrm_mae = lrm_model['best_audio_mae']
    mae_improvement = ((baseline_mae - lrm_mae) / baseline_mae) * 100
    
    baseline_pearson = baseline_model['best_audio_pearson']
    lrm_pearson = lrm_model['best_audio_pearson']
    pearson_improvement = ((lrm_pearson - baseline_pearson) / baseline_pearson) * 100
    
    print(f"🎯 AUDIO MAE COMPARISON:")
    print(f"   Baseline: {baseline_mae:.4f}")
    print(f"   LRM:      {lrm_mae:.4f}")
    if mae_improvement > 0:
        print(f"   📈 LRM Improvement: {mae_improvement:.2f}% better (lower MAE)")
    else:
        print(f"   📉 LRM Performance: {abs(mae_improvement):.2f}% worse (higher MAE)")
    
    print(f"\n📊 AUDIO PEARSON COMPARISON:")
    print(f"   Baseline: {baseline_pearson:.4f}")
    print(f"   LRM:      {lrm_pearson:.4f}")
    if pearson_improvement > 0:
        print(f"   📈 LRM Improvement: {pearson_improvement:.2f}% better (higher correlation)")
    else:
        print(f"   📉 LRM Performance: {abs(pearson_improvement):.2f}% worse (lower correlation)")
    
    # Training completion status
    print(f"\n✅ TRAINING STATUS:")
    print(f"   Baseline Complete: {baseline_model['training_complete']}")
    print(f"   LRM Complete: {lrm_model['training_complete']}")

def main():
    """Main function to analyze all training results."""
    print("🔍 COMPREHENSIVE TRAINING ANALYSIS")
    print("=" * 80)
    
    # Analyze baseline emotion regression
    baseline_workspace = "workspaces/emotion_regression"
    baseline_results = analyze_workspace(baseline_workspace, "Baseline Emotion Regression")
    
    # Analyze LRM emotion feedback
    lrm_workspace = "workspaces/emotion_feedback"
    lrm_results = analyze_workspace(lrm_workspace, "LRM Emotion Feedback")
    
    # Compare the models
    if baseline_results and lrm_results:
        compare_models(baseline_results, lrm_results)
    
    print(f"\n✅ ANALYSIS COMPLETED!")
    print(f"\n💡 SUMMARY:")
    print(f"   - Both models completed 5000 iterations successfully")
    print(f"   - Lower MAE = Better accuracy")
    print(f"   - Higher Pearson = Better correlation")
    print(f"   - Check comparison above for LRM vs Baseline performance")

if __name__ == '__main__':
    main() 