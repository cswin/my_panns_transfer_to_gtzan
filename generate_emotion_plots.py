#!/usr/bin/env python3
"""
Standalone script to generate emotion prediction visualizations.

Usage:
    python generate_emotion_plots.py /path/to/predictions/directory

This script reads segment_predictions.csv and audio_predictions.csv from the specified directory
and generates comprehensive visualizations including scatter plots and time-series plots.
"""

import sys
import os

# Add the current directory to sys.path so we can import emotion_visualize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from emotion_visualize import create_emotion_visualizations
    
    def main():
        if len(sys.argv) != 2:
            print("Usage: python generate_emotion_plots.py <predictions_directory>")
            print("\nThe predictions directory should contain:")
            print("- segment_predictions.csv")
            print("- audio_predictions.csv")
            sys.exit(1)
        
        predictions_dir = sys.argv[1]
        
        if not os.path.exists(predictions_dir):
            print(f"Error: Directory {predictions_dir} does not exist")
            sys.exit(1)
        
        # Check if required CSV files exist
        segment_csv = os.path.join(predictions_dir, 'segment_predictions.csv')
        audio_csv = os.path.join(predictions_dir, 'audio_predictions.csv')
        
        if not os.path.exists(segment_csv):
            print(f"Error: segment_predictions.csv not found in {predictions_dir}")
            sys.exit(1)
        
        if not os.path.exists(audio_csv):
            print(f"Error: audio_predictions.csv not found in {predictions_dir}")
            sys.exit(1)
        
        print(f"Generating emotion visualizations from: {predictions_dir}")
        create_emotion_visualizations(predictions_dir)
        print("Visualization generation completed!")
    
    if __name__ == '__main__':
        main()

except ImportError as e:
    print(f"Error importing visualization module: {e}")
    print("Please ensure you have installed all required dependencies:")
    print("pip install matplotlib seaborn pandas numpy scipy scikit-learn")
    sys.exit(1) 