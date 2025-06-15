#!/usr/bin/env python3

import sys
import os
sys.path.append('./utils')
from features import create_indexes

def main():
    # Path to the main features file (use absolute path for remote machine)
    features_path = '/home/pengliu/Private/panns_transfer_to_gtzan/features/features.h5'
    
    if not os.path.exists(features_path):
        print(f"Error: Features file not found at {features_path}")
        print("Make sure the feature extraction completed successfully.")
        print("Available files in features directory:")
        features_dir = os.path.dirname(features_path)
        if os.path.exists(features_dir):
            for f in os.listdir(features_dir):
                print(f"  {f}")
        return
    
    print("Creating train and validation index files...")
    print(f"Source: {features_path}")
    
    # Create the index files with 70/30 split
    create_indexes(features_path, train_ratio=0.7)
    
    print("Index files created successfully!")
    print("You can now run the training script.")

if __name__ == '__main__':
    main() 