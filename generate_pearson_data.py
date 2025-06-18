#!/usr/bin/env python3
"""
Generate separate valence and arousal Pearson correlation data from existing statistics.
"""

import os
import pickle
import numpy as np
from scipy.stats import pearsonr
import glob

def load_statistics_pickle(workspace_path, model_name):
    """Load the most recent statistics file for a model."""
    stats_pattern = os.path.join(workspace_path, "statistics", "**", f"*{model_name}*", "**", "statistics*.pkl")
    stats_files = glob.glob(stats_pattern, recursive=True)
    
    if not stats_files:
        return None
    
    # Sort by modification time and get the most recent
    stats_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_stats = stats_files[0]
    
    print(f"Loading statistics from: {os.path.basename(latest_stats)}")
    
    try:
        with open(latest_stats, 'rb') as f:
            statistics_container = pickle.load(f)
        return statistics_container
    except Exception as e:
        print(f"Error loading statistics: {e}")
        return None

def extract_pearson_correlations(statistics_container):
    """Extract valence and arousal Pearson correlations from statistics."""
    iterations = []
    val_valence_pearson = []
    val_arousal_pearson = []
    
    if hasattr(statistics_container, 'statistics_dict'):
        for data_type, data in statistics_container.statistics_dict.items():
            if data_type == 'validate':
                for iteration, stats in data.items():
                    if 'audio_valence_pearson' in stats and 'audio_arousal_pearson' in stats:
                        iterations.append(iteration)
                        val_valence_pearson.append(stats['audio_valence_pearson'])
                        val_arousal_pearson.append(stats['audio_arousal_pearson'])
    
    return {
        'iterations': iterations,
        'val_valence_pearson': val_valence_pearson,
        'val_arousal_pearson': val_arousal_pearson
    }

def append_to_log_file(log_path, pearson_data):
    """Append separate Pearson correlation entries to log file."""
    print(f"Appending Pearson data to: {os.path.basename(log_path)}")
    
    if not pearson_data['iterations']:
        print("No Pearson data found in statistics")
        return False
    
    try:
        with open(log_path, 'a') as f:
            for i, iteration in enumerate(pearson_data['iterations']):
                valence_pearson = pearson_data['val_valence_pearson'][i]
                arousal_pearson = pearson_data['val_arousal_pearson'][i]
                
                # Add fake log entry with the separate correlations
                f.write(f"\n# Generated entry - Iteration: {iteration}\n")
                f.write(f"INFO Validate Audio Valence Pearson: {valence_pearson:.4f}, Arousal Pearson: {arousal_pearson:.4f}\n")
        return True
    except Exception as e:
        print(f"Error writing to log file: {e}")
        return False

def main():
    """Generate separate Pearson correlation data for plotting."""
    print("📈 GENERATING SEPARATE PEARSON CORRELATION DATA")
    print("=" * 50)
    
    # Load statistics for both models
    baseline_stats = load_statistics_pickle("workspaces/emotion_regression", "FeatureEmotionRegression_Cnn6_NewAffective")
    lrm_stats = load_statistics_pickle("workspaces/emotion_feedback", "FeatureEmotionRegression_Cnn6_LRM")
    
    if not baseline_stats or not lrm_stats:
        print("❌ Could not load statistics files")
        return
    
    # Extract Pearson correlations
    baseline_pearson = extract_pearson_correlations(baseline_stats)
    lrm_pearson = extract_pearson_correlations(lrm_stats)
    
    # Find log files
    baseline_log_pattern = "workspaces/emotion_regression/logs/**/*FeatureEmotionRegression_Cnn6_NewAffective*/**/*.log"
    lrm_log_pattern = "workspaces/emotion_feedback/logs/**/*FeatureEmotionRegression_Cnn6_LRM*/**/*.log"
    
    baseline_logs = glob.glob(baseline_log_pattern, recursive=True)
    lrm_logs = glob.glob(lrm_log_pattern, recursive=True)
    
    if baseline_logs:
        baseline_log = max(baseline_logs, key=lambda x: os.path.getmtime(x))
        if append_to_log_file(baseline_log, baseline_pearson):
            print(f"✅ Updated baseline log with {len(baseline_pearson['iterations'])} Pearson entries")
    
    if lrm_logs:
        lrm_log = max(lrm_logs, key=lambda x: os.path.getmtime(x))
        if append_to_log_file(lrm_log, lrm_pearson):
            print(f"✅ Updated LRM log with {len(lrm_pearson['iterations'])} Pearson entries")
    
    print(f"\n🎯 Now you can run: python plot_training_curves.py")
    print(f"   This will generate all 4 separate plots you requested!")

if __name__ == '__main__':
    main() 