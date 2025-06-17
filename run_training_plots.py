#!/usr/bin/env python3
"""
Simple script to run training performance comparison plots.

This script demonstrates how to use the training comparison plotter
for both run_emotion.sh and run_emotion_feedback.sh results.
"""

import os
import sys
from plot_training_comparison import TrainingLogParser, TrainingPlotter

def demo_with_sample_data():
    """Create demo plots with sample training data."""
    print("📊 Creating demo training comparison plots...")
    
    # Sample baseline data (simulating typical training curve)
    baseline_data = {
        'epochs': list(range(1, 21)),
        'train_loss': [2.5, 2.1, 1.8, 1.6, 1.4, 1.3, 1.2, 1.1, 1.0, 0.95, 
                      0.9, 0.87, 0.85, 0.83, 0.81, 0.8, 0.79, 0.78, 0.77, 0.76],
        'val_loss': [2.6, 2.2, 1.9, 1.7, 1.5, 1.4, 1.3, 1.2, 1.1, 1.05, 
                    1.0, 0.98, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.9, 0.89],
        'val_mae': [0.85, 0.78, 0.72, 0.68, 0.65, 0.62, 0.6, 0.58, 0.56, 0.55,
                   0.54, 0.53, 0.52, 0.51, 0.5, 0.49, 0.48, 0.47, 0.46, 0.45],
        'val_pearson': [0.3, 0.35, 0.4, 0.45, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6,
                       0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71]
    }
    
    # Sample LRM data (simulating better performance with feedback)
    lrm_data = {
        'epochs': list(range(1, 21)),
        'train_loss': [2.4, 2.0, 1.7, 1.4, 1.2, 1.0, 0.9, 0.8, 0.75, 0.7, 
                      0.65, 0.62, 0.6, 0.58, 0.56, 0.54, 0.52, 0.5, 0.48, 0.46],
        'val_loss': [2.5, 2.1, 1.8, 1.5, 1.3, 1.1, 1.0, 0.9, 0.85, 0.8, 
                    0.75, 0.72, 0.7, 0.68, 0.66, 0.64, 0.62, 0.6, 0.58, 0.56],
        'val_mae': [0.82, 0.74, 0.67, 0.61, 0.56, 0.52, 0.48, 0.45, 0.42, 0.4,
                   0.38, 0.36, 0.34, 0.32, 0.3, 0.28, 0.26, 0.24, 0.22, 0.2],
        'val_pearson': [0.35, 0.42, 0.48, 0.54, 0.6, 0.65, 0.68, 0.71, 0.74, 0.76,
                       0.78, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88]
    }
    
    # Create plotter and generate plots
    plotter = TrainingPlotter()
    plotter.plot_comparison(baseline_data, lrm_data, './demo_training_plots')
    
    print("✅ Demo plots created in: ./demo_training_plots")
    print("\n📈 Demo Results Summary:")
    print("-" * 40)
    print(f"Baseline final validation MAE: {baseline_data['val_mae'][-1]:.3f}")
    print(f"LRM final validation MAE: {lrm_data['val_mae'][-1]:.3f}")
    
    improvement = ((baseline_data['val_mae'][-1] - lrm_data['val_mae'][-1]) / baseline_data['val_mae'][-1]) * 100
    print(f"MAE improvement: {improvement:+.1f}%")
    
    print(f"Baseline final Pearson correlation: {baseline_data['val_pearson'][-1]:.3f}")
    print(f"LRM final Pearson correlation: {lrm_data['val_pearson'][-1]:.3f}")
    
    corr_improvement = ((lrm_data['val_pearson'][-1] - baseline_data['val_pearson'][-1]) / baseline_data['val_pearson'][-1]) * 100
    print(f"Correlation improvement: {corr_improvement:+.1f}%")

def find_and_plot_real_logs():
    """Find and plot real training logs if available."""
    print("🔍 Searching for real training logs...")
    
    # Common log locations
    log_locations = [
        './workspaces/emotion_regression/logs/',
        './logs/',
        './checkpoints/',
        './'
    ]
    
    found_logs = []
    for location in log_locations:
        if os.path.exists(location):
            for file in os.listdir(location):
                if file.endswith('.log') or 'log' in file.lower():
                    found_logs.append(os.path.join(location, file))
    
    if found_logs:
        print(f"Found {len(found_logs)} log files:")
        for log in found_logs:
            print(f"  - {log}")
        
        # Try to parse and plot
        parser = TrainingLogParser()
        plotter = TrainingPlotter()
        
        baseline_data = {}
        lrm_data = {}
        
        for log_file in found_logs:
            try:
                data = parser.parse_log_file(log_file)
                if data['epochs'] or data['iterations']:
                    if 'lrm' in log_file.lower() or 'feedback' in log_file.lower():
                        lrm_data = data
                        print(f"✅ Parsed LRM log: {log_file}")
                    else:
                        baseline_data = data
                        print(f"✅ Parsed baseline log: {log_file}")
            except Exception as e:
                print(f"❌ Failed to parse {log_file}: {e}")
        
        if baseline_data or lrm_data:
            plotter.plot_comparison(baseline_data, lrm_data, './real_training_plots')
            print("✅ Real training plots created in: ./real_training_plots")
        else:
            print("❌ No valid training data found in logs")
    else:
        print("❌ No log files found")

def main():
    print("🚀 Training Performance Comparison Tool")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--demo':
            demo_with_sample_data()
        elif sys.argv[1] == '--real':
            find_and_plot_real_logs()
        elif sys.argv[1] == '--help':
            print_help()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print_help()
    else:
        print("🎯 Running both demo and real log search...")
        print("\n1. Creating demo plots...")
        demo_with_sample_data()
        
        print("\n2. Searching for real logs...")
        find_and_plot_real_logs()

def print_help():
    print("""
Usage: python run_training_plots.py [OPTIONS]

Options:
  --demo    Create demo plots with sample data
  --real    Search for and plot real training logs
  --help    Show this help message

Examples:
  python run_training_plots.py              # Run both demo and real
  python run_training_plots.py --demo       # Demo only
  python run_training_plots.py --real       # Real logs only

For more advanced usage, use plot_training_comparison.py directly:
  python plot_training_comparison.py --auto-find
  python plot_training_comparison.py --baseline-logs baseline.log --lrm-logs lrm.log
""")

if __name__ == '__main__':
    main() 