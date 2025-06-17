#!/usr/bin/env python3
"""
Training Performance Comparison Plotter

This script generates training performance plots comparing baseline and LRM models.
It can parse different log formats and create comprehensive visualizations.

Usage:
    python plot_training_comparison.py --baseline_logs <path> --lrm_logs <path>
    python plot_training_comparison.py --auto-find  # Automatically find log files
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob

# Optional pandas import
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

class TrainingLogParser:
    """Parse different types of training logs."""
    
    def __init__(self):
        self.log_patterns = {
            # Pattern for standard emotion_main.py logs
            'emotion_main': {
                'iteration': r'Iteration: (\d+)',
                'val_mae': r'Validate Audio Mean MAE: ([\d.]+)',
                'val_rmse': r'Validate Audio Mean RMSE: ([\d.]+)',
                'val_pearson': r'Validate Audio Mean Pearson: ([\d.]+)',
                'val_valence_mae': r'Validate Audio Valence MAE: ([\d.]+)',
                'val_arousal_mae': r'Validate Audio Arousal MAE: ([\d.]+)',
                'train_loss': r'loss: ([\d.]+)'
            },
            
            # Pattern for test scripts with epoch-based training
            'epoch_based': {
                'epoch': r'Epoch (\d+)',
                'train_loss': r'Training: Loss=([\d.]+)',
                'val_loss': r'Validation: Loss=([\d.]+)',
                'train_valence': r'Training:.*V=([\d.]+)',
                'train_arousal': r'Training:.*A=([\d.]+)',
                'val_valence': r'Validation:.*V=([\d.]+)',
                'val_arousal': r'Validation:.*A=([\d.]+)'
            }
        }
    
    def detect_log_format(self, log_path: str) -> str:
        """Detect the format of the log file."""
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Check for epoch-based format
        if 'Epoch' in content and 'Training:' in content:
            return 'epoch_based'
        
        # Default to emotion_main format
        return 'emotion_main'
    
    def parse_log_file(self, log_path: str) -> Dict:
        """Parse a log file and extract training metrics."""
        format_type = self.detect_log_format(log_path)
        return self._parse_text_log(log_path, format_type)
    
    def _parse_text_log(self, log_path: str, format_type: str) -> Dict:
        """Parse text format logs."""
        patterns = self.log_patterns[format_type]
        
        data = {
            'epochs': [],
            'iterations': [],
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'val_rmse': [],
            'val_pearson': [],
            'val_valence_mae': [],
            'val_arousal_mae': []
        }
        
        with open(log_path, 'r') as f:
            content = f.read()
        
        # Extract data based on format
        if format_type == 'emotion_main':
            self._parse_emotion_main_format(content, patterns, data)
        elif format_type == 'epoch_based':
            self._parse_epoch_based_format(content, patterns, data)
        
        return data
    
    def _parse_emotion_main_format(self, content: str, patterns: Dict, data: Dict):
        """Parse emotion_main.py log format."""
        lines = content.split('\n')
        current_iteration = None
        
        for line in lines:
            # Extract iteration and training loss
            if 'Iteration:' in line and 'loss:' in line:
                iter_match = re.search(patterns['iteration'], line)
                loss_match = re.search(patterns['train_loss'], line)
                if iter_match and loss_match:
                    current_iteration = int(iter_match.group(1))
                    data['iterations'].append(current_iteration)
                    data['train_loss'].append(float(loss_match.group(1)))
            
            # Extract validation metrics
            elif current_iteration and 'Validate' in line:
                for metric, pattern in patterns.items():
                    if metric.startswith('val_'):
                        match = re.search(pattern, line)
                        if match:
                            # Ensure we have the same number of entries
                            while len(data[metric]) < len(data['iterations']) - 1:
                                data[metric].append(None)
                            data[metric].append(float(match.group(1)))
        
        # Convert iterations to epochs (assuming 200 iterations per evaluation)
        data['epochs'] = [i // 200 for i in data['iterations']]
    
    def _parse_epoch_based_format(self, content: str, patterns: Dict, data: Dict):
        """Parse epoch-based log format."""
        lines = content.split('\n')
        
        for line in lines:
            # Extract epoch number
            epoch_match = re.search(patterns['epoch'], line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                
                # Extract training metrics
                if 'Training:' in line:
                    data['epochs'].append(epoch)
                    
                    train_loss_match = re.search(patterns['train_loss'], line)
                    if train_loss_match:
                        data['train_loss'].append(float(train_loss_match.group(1)))
                
                # Extract validation metrics
                elif 'Validation:' in line:
                    val_loss_match = re.search(patterns['val_loss'], line)
                    if val_loss_match:
                        data['val_loss'].append(float(val_loss_match.group(1)))


class TrainingPlotter:
    """Create training performance plots."""
    
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        plt.style.use('default')
        # Set up nice colors
        self.colors = {
            'baseline': '#2E86AB',  # Blue
            'lrm': '#A23B72',       # Purple
            'train': '#F18F01',     # Orange
            'val': '#C73E1D'        # Red
        }
    
    def plot_comparison(self, baseline_data: Dict, lrm_data: Dict, output_dir: str = './plots'):
        """Create comprehensive comparison plots."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create main comparison plot
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Training Performance Comparison: Baseline vs LRM', fontsize=16, fontweight='bold')
        
        # Plot 1: Training Loss
        self._plot_loss_comparison(axes[0, 0], baseline_data, lrm_data, 'train_loss', 'Training Loss')
        
        # Plot 2: Validation Loss
        self._plot_loss_comparison(axes[0, 1], baseline_data, lrm_data, 'val_loss', 'Validation Loss')
        
        # Plot 3: Validation MAE
        self._plot_loss_comparison(axes[1, 0], baseline_data, lrm_data, 'val_mae', 'Validation MAE')
        
        # Plot 4: Validation Pearson Correlation
        self._plot_loss_comparison(axes[1, 1], baseline_data, lrm_data, 'val_pearson', 'Validation Pearson Correlation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create detailed plots
        self._create_detailed_plots(baseline_data, lrm_data, output_dir)
    
    def _plot_loss_comparison(self, ax, baseline_data: Dict, lrm_data: Dict, metric: str, title: str):
        """Plot comparison for a specific metric."""
        # Plot baseline
        if metric in baseline_data and baseline_data[metric]:
            baseline_values = [v for v in baseline_data[metric] if v is not None]
            if baseline_values:
                baseline_epochs = list(range(1, len(baseline_values) + 1))
                ax.plot(baseline_epochs, baseline_values, 
                       color=self.colors['baseline'], linewidth=2, 
                       label='Baseline', marker='o', markersize=4)
        
        # Plot LRM
        if metric in lrm_data and lrm_data[metric]:
            lrm_values = [v for v in lrm_data[metric] if v is not None]
            if lrm_values:
                lrm_epochs = list(range(1, len(lrm_values) + 1))
                ax.plot(lrm_epochs, lrm_values, 
                       color=self.colors['lrm'], linewidth=2, 
                       label='LRM', marker='s', markersize=4)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_detailed_plots(self, baseline_data: Dict, lrm_data: Dict, output_dir: str):
        """Create detailed individual plots."""
        
        # Separate plots for each metric
        metrics_to_plot = [
            ('train_loss', 'Training Loss Over Epochs'),
            ('val_loss', 'Validation Loss Over Epochs'),
            ('val_mae', 'Validation MAE Over Epochs'),
            ('val_rmse', 'Validation RMSE Over Epochs'),
            ('val_pearson', 'Validation Pearson Correlation Over Epochs')
        ]
        
        for metric, title in metrics_to_plot:
            if (metric in baseline_data and baseline_data[metric]) or \
               (metric in lrm_data and lrm_data[metric]):
                
                plt.figure(figsize=(10, 6))
                
                # Plot baseline
                if metric in baseline_data and baseline_data[metric]:
                    baseline_values = [v for v in baseline_data[metric] if v is not None]
                    if baseline_values:
                        baseline_epochs = list(range(1, len(baseline_values) + 1))
                        plt.plot(baseline_epochs, baseline_values, 
                               color=self.colors['baseline'], linewidth=2, 
                               label='Baseline', marker='o', markersize=6)
                
                # Plot LRM
                if metric in lrm_data and lrm_data[metric]:
                    lrm_values = [v for v in lrm_data[metric] if v is not None]
                    if lrm_values:
                        lrm_epochs = list(range(1, len(lrm_values) + 1))
                        plt.plot(lrm_epochs, lrm_values, 
                               color=self.colors['lrm'], linewidth=2, 
                               label='LRM', marker='s', markersize=6)
                
                plt.title(title, fontsize=14, fontweight='bold')
                plt.xlabel('Epoch', fontsize=12)
                plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
                plt.legend(fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                filename = f"{metric}_comparison.png"
                plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
                plt.close()


def find_log_files(search_dirs: List[str] = None) -> Dict[str, List[str]]:
    """Automatically find training log files."""
    if search_dirs is None:
        search_dirs = ['.', './workspaces', './logs', './checkpoints']
    
    log_files = {'baseline': [], 'lrm': []}
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            # Look for various log file patterns
            patterns = [
                '**/*train*.log',
                '**/*emotion*.log', 
                '**/*log*',
                '**/train.log',
                '**/training.log'
            ]
            
            for pattern in patterns:
                for log_file in glob.glob(os.path.join(search_dir, pattern), recursive=True):
                    # Categorize based on filename or directory
                    if any(keyword in log_file.lower() for keyword in ['lrm', 'feedback']):
                        log_files['lrm'].append(log_file)
                    else:
                        log_files['baseline'].append(log_file)
    
    return log_files


def main():
    parser = argparse.ArgumentParser(description='Plot training performance comparison')
    parser.add_argument('--baseline-logs', type=str, help='Path to baseline training logs')
    parser.add_argument('--lrm-logs', type=str, help='Path to LRM training logs')
    parser.add_argument('--auto-find', action='store_true', help='Automatically find log files')
    parser.add_argument('--output-dir', type=str, default='./training_plots', help='Output directory for plots')
    parser.add_argument('--search-dirs', nargs='+', help='Directories to search for logs')
    
    args = parser.parse_args()
    
    # Initialize parser and plotter
    log_parser = TrainingLogParser()
    plotter = TrainingPlotter()
    
    baseline_data = {}
    lrm_data = {}
    
    if args.auto_find:
        print("🔍 Auto-finding log files...")
        log_files = find_log_files(args.search_dirs)
        
        print(f"Found baseline logs: {log_files['baseline']}")
        print(f"Found LRM logs: {log_files['lrm']}")
        
        # Parse baseline logs
        for log_file in log_files['baseline']:
            try:
                data = log_parser.parse_log_file(log_file)
                if data['epochs'] or data['iterations']:
                    baseline_data = data
                    print(f"✅ Parsed baseline log: {log_file}")
                    break
            except Exception as e:
                print(f"❌ Failed to parse {log_file}: {e}")
        
        # Parse LRM logs
        for log_file in log_files['lrm']:
            try:
                data = log_parser.parse_log_file(log_file)
                if data['epochs'] or data['iterations']:
                    lrm_data = data
                    print(f"✅ Parsed LRM log: {log_file}")
                    break
            except Exception as e:
                print(f"❌ Failed to parse {log_file}: {e}")
    
    else:
        # Parse specified log files
        if args.baseline_logs:
            print(f"📊 Parsing baseline logs: {args.baseline_logs}")
            baseline_data = log_parser.parse_log_file(args.baseline_logs)
        
        if args.lrm_logs:
            print(f"📊 Parsing LRM logs: {args.lrm_logs}")
            lrm_data = log_parser.parse_log_file(args.lrm_logs)
    
    # Create plots
    if baseline_data or lrm_data:
        print(f"📈 Creating training comparison plots...")
        plotter.plot_comparison(baseline_data, lrm_data, args.output_dir)
        print(f"✅ Plots saved in: {args.output_dir}")
        
        # Print summary statistics
        print("\n📊 Training Summary:")
        print("-" * 40)
        
        if baseline_data.get('train_loss'):
            final_loss = baseline_data['train_loss'][-1] if baseline_data['train_loss'] else 'N/A'
            print(f"Baseline final training loss: {final_loss}")
        
        if lrm_data.get('train_loss'):
            final_loss = lrm_data['train_loss'][-1] if lrm_data['train_loss'] else 'N/A'
            print(f"LRM final training loss: {final_loss}")
        
        if baseline_data.get('val_mae') and lrm_data.get('val_mae'):
            baseline_mae = baseline_data['val_mae'][-1] if baseline_data['val_mae'] else float('inf')
            lrm_mae = lrm_data['val_mae'][-1] if lrm_data['val_mae'] else float('inf')
            if baseline_mae != float('inf') and lrm_mae != float('inf'):
                improvement = ((baseline_mae - lrm_mae) / baseline_mae) * 100
                print(f"MAE improvement: {improvement:+.1f}% (Baseline: {baseline_mae:.4f}, LRM: {lrm_mae:.4f})")
    
    else:
        print("❌ No valid training logs found. Please check your log file paths.")
        print("\nUsage examples:")
        print("  python plot_training_comparison.py --auto-find")
        print("  python plot_training_comparison.py --baseline-logs baseline.log --lrm-logs lrm.log")


if __name__ == '__main__':
    main() 