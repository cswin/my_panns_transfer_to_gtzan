#!/usr/bin/env python3

import os
import glob
import re
from datetime import datetime

def parse_log_file(log_path):
    """Parse a training log file and extract key metrics."""
    if not os.path.exists(log_path):
        return None
    
    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_pearson': [],
        'last_epoch': 0,
        'best_mae': float('inf'),
        'best_pearson': -1.0,
        'training_complete': False
    }
    
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Look for epoch information
            if 'Epoch:' in line and 'Train loss:' in line:
                # Extract epoch number
                epoch_match = re.search(r'Epoch:\s*(\d+)', line)
                if epoch_match:
                    epoch = int(epoch_match.group(1))
                    metrics['epochs'].append(epoch)
                    metrics['last_epoch'] = max(metrics['last_epoch'], epoch)
                
                # Extract train loss
                train_loss_match = re.search(r'Train loss:\s*([\d.]+)', line)
                if train_loss_match:
                    metrics['train_loss'].append(float(train_loss_match.group(1)))
            
            # Look for validation metrics
            if 'Audio Mean MAE:' in line:
                mae_match = re.search(r'Audio Mean MAE:\s*([\d.]+)', line)
                if mae_match:
                    mae = float(mae_match.group(1))
                    metrics['val_mae'].append(mae)
                    metrics['best_mae'] = min(metrics['best_mae'], mae)
            
            if 'Audio Mean Pearson:' in line:
                pearson_match = re.search(r'Audio Mean Pearson:\s*([\d.]+)', line)
                if pearson_match:
                    pearson = float(pearson_match.group(1))
                    metrics['val_pearson'].append(pearson)
                    metrics['best_pearson'] = max(metrics['best_pearson'], pearson)
            
            # Check if training completed
            if 'Training completed' in line or 'Best validation' in line:
                metrics['training_complete'] = True
    
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        return None
    
    return metrics

def check_workspace_logs(workspace_path, model_name):
    """Check logs in a specific workspace directory."""
    print(f"\n🔍 Checking {model_name} workspace: {workspace_path}")
    
    if not os.path.exists(workspace_path):
        print(f"   ❌ Workspace doesn't exist: {workspace_path}")
        return
    
    # Check for logs directory
    logs_dir = os.path.join(workspace_path, 'logs')
    if not os.path.exists(logs_dir):
        print(f"   ❌ No logs directory found")
        return
    
    # Find log files
    log_files = glob.glob(os.path.join(logs_dir, '*.log'))
    if not log_files:
        print(f"   ❌ No log files found")
        return
    
    print(f"   📄 Found {len(log_files)} log file(s):")
    for log_file in log_files:
        print(f"      - {os.path.basename(log_file)}")
    
    # Parse the main training log
    main_log = None
    for log_file in log_files:
        if 'train' in os.path.basename(log_file).lower():
            main_log = log_file
            break
    
    if not main_log and log_files:
        main_log = log_files[0]  # Use first log file if no train log found
    
    if main_log:
        print(f"   📊 Analyzing: {os.path.basename(main_log)}")
        metrics = parse_log_file(main_log)
        
        if metrics:
            print(f"      ✅ Last epoch: {metrics['last_epoch']}")
            print(f"      ✅ Training complete: {metrics['training_complete']}")
            
            if metrics['best_mae'] != float('inf'):
                print(f"      📈 Best MAE: {metrics['best_mae']:.4f}")
            
            if metrics['best_pearson'] != -1.0:
                print(f"      📈 Best Pearson: {metrics['best_pearson']:.4f}")
            
            if metrics['train_loss']:
                print(f"      📉 Final train loss: {metrics['train_loss'][-1]:.4f}")
        else:
            print(f"      ❌ Could not parse log file")
    
    # Check for checkpoints
    checkpoints_dir = os.path.join(workspace_path, 'checkpoints')
    if os.path.exists(checkpoints_dir):
        checkpoints = glob.glob(os.path.join(checkpoints_dir, '*.pth'))
        print(f"   💾 Found {len(checkpoints)} checkpoint(s)")
        for checkpoint in checkpoints:
            print(f"      - {os.path.basename(checkpoint)}")
    else:
        print(f"   ❌ No checkpoints directory found")
    
    # Check for statistics
    stats_dir = os.path.join(workspace_path, 'statistics')
    if os.path.exists(stats_dir):
        stats_files = glob.glob(os.path.join(stats_dir, '*.pkl'))
        print(f"   📊 Found {len(stats_files)} statistics file(s)")
        for stats_file in stats_files:
            print(f"      - {os.path.basename(stats_file)}")
    else:
        print(f"   ❌ No statistics directory found")

def main():
    """Main function to check all training logs."""
    print("🔍 Checking Training Logs")
    print("=" * 50)
    
    # Check baseline emotion regression
    baseline_workspace = "workspaces/emotion_regression"
    check_workspace_logs(baseline_workspace, "Baseline Emotion Regression")
    
    # Check LRM emotion feedback
    lrm_workspace = "workspaces/emotion_feedback"
    check_workspace_logs(lrm_workspace, "LRM Emotion Feedback")
    
    # Check for any other workspaces
    workspaces_dir = "workspaces"
    if os.path.exists(workspaces_dir):
        all_workspaces = [d for d in os.listdir(workspaces_dir) 
                         if os.path.isdir(os.path.join(workspaces_dir, d))]
        
        other_workspaces = [w for w in all_workspaces 
                           if w not in ['emotion_regression', 'emotion_feedback']]
        
        for workspace in other_workspaces:
            workspace_path = os.path.join(workspaces_dir, workspace)
            check_workspace_logs(workspace_path, f"Other ({workspace})")
    
    print(f"\n✅ Log check completed!")
    print(f"\n💡 Tips:")
    print(f"   - Look for 'Audio Mean MAE' and 'Audio Mean Pearson' as key metrics")
    print(f"   - Lower MAE is better (closer to 0)")
    print(f"   - Higher Pearson correlation is better (closer to 1)")
    print(f"   - Compare baseline vs LRM performance")

if __name__ == '__main__':
    main() 