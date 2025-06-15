#!/usr/bin/env python3

import os
import glob
from collections import defaultdict

def check_dataset_files():
    """Check what files are actually in the GTZAN dataset directories."""
    
    dataset_dir = "/DATA/pliu/EmotionData/GTZAN/genres_original/"
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    print("=== GTZAN Dataset File Analysis ===")
    print(f"Dataset directory: {dataset_dir}")
    print()
    
    total_files = 0
    file_extensions = defaultdict(int)
    
    for genre in genres:
        genre_dir = os.path.join(dataset_dir, genre)
        if not os.path.exists(genre_dir):
            print(f"❌ {genre}: Directory missing")
            continue
            
        # Get all files in the genre directory
        all_files = os.listdir(genre_dir)
        audio_files = [f for f in all_files if not f.startswith('.')]  # Skip hidden files
        
        print(f"📁 {genre}: {len(audio_files)} files")
        
        # Check file extensions
        for filename in audio_files:
            ext = os.path.splitext(filename)[1].lower()
            file_extensions[ext] += 1
        
        # Show first few files as examples
        if audio_files:
            print(f"   Examples: {audio_files[:3]}")
        
        total_files += len(audio_files)
    
    print(f"\n=== File Extension Summary ===")
    for ext, count in sorted(file_extensions.items()):
        print(f"{ext}: {count} files")
    
    print(f"\nTotal files found: {total_files}")
    print(f"Expected: 1000 files")
    print(f"In features.h5: 325 files")
    print(f"Missing during processing: {total_files - 325} files")
    
    # Check what the feature extraction script expects
    print(f"\n=== Feature Extraction Filter Check ===")
    print("The feature extraction script filters files with this logic:")
    print("1. Filename must not be empty")
    print("2. Filename must contain a genre prefix")
    print("3. Genre prefix must be in the labels list")
    
    # Simulate the filtering logic
    lb_to_idx = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 
                 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
    
    filtered_files = []
    for genre in genres:
        genre_dir = os.path.join(dataset_dir, genre)
        if not os.path.exists(genre_dir):
            continue
            
        for filename in os.listdir(genre_dir):
            if filename.startswith('.'):
                continue
                
            file_path = os.path.join(genre_dir, filename)
            
            # Apply the same filter logic as features.py
            if filename and filename.split('.')[0] and filename.split('.')[0] in lb_to_idx:
                filtered_files.append((filename, file_path))
            else:
                print(f"❌ Filtered out: {filename} (genre: {filename.split('.')[0]})")
    
    print(f"\nFiles that would pass the filter: {len(filtered_files)}")
    
    if len(filtered_files) != 325:
        print(f"⚠️  Filter mismatch: Expected {len(filtered_files)}, but got 325 in features.h5")
        print("This suggests additional filtering happened during librosa loading")

if __name__ == '__main__':
    check_dataset_files() 