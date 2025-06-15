#!/usr/bin/env python3
"""
Script to flatten the Emo-Soundscapes audio directory structure.

This script copies all audio files from:
- 600_Sounds/* subdirectories
- 613_MixedSounds/* subdirectories
Into a single flat directory for easier feature extraction.
"""

import os
import shutil
import glob
import argparse
from pathlib import Path

def flatten_audio_directory(source_dirs, output_dir, copy_files=True):
    """
    Flatten audio directory structure by copying all audio files to one directory.
    
    Args:
        source_dirs: list of source directories containing subdirectories with audio files
        output_dir: output directory to copy all files to
        copy_files: if True, copy files; if False, just list them
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Audio file extensions to look for
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.aac']
    
    total_files = 0
    processed_files = 0
    
    # Process each source directory
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory does not exist: {source_dir}")
            continue
            
        print(f"\nProcessing: {source_dir}")
        
        # Find all audio files in subdirectories
        audio_files = []
        for ext in audio_extensions:
            # Search recursively for audio files
            pattern = os.path.join(source_dir, '**', ext)
            audio_files.extend(glob.glob(pattern, recursive=True))
        
        print(f"Found {len(audio_files)} audio files")
        total_files += len(audio_files)
        
        # Copy each file to the output directory
        for audio_file in audio_files:
            filename = os.path.basename(audio_file)
            output_path = os.path.join(output_dir, filename)
            
            # Check for filename conflicts
            if os.path.exists(output_path):
                print(f"Warning: File already exists, skipping: {filename}")
                continue
            
            if copy_files:
                try:
                    shutil.copy2(audio_file, output_path)
                    processed_files += 1
                    if processed_files % 100 == 0:
                        print(f"Processed {processed_files}/{total_files} files...")
                except Exception as e:
                    print(f"Error copying {audio_file}: {e}")
            else:
                print(f"Would copy: {audio_file} -> {output_path}")
                processed_files += 1
    
    print(f"\n=== Summary ===")
    print(f"Total files found: {total_files}")
    print(f"Files processed: {processed_files}")
    
    if copy_files:
        print(f"All audio files copied to: {output_dir}")
    else:
        print("Dry run completed. Use --copy to actually copy files.")
        
    return processed_files

def check_ratings_coverage(audio_dir, ratings_dir):
    """
    Check how many audio files have corresponding ratings.
    """
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not installed, skipping ratings coverage check")
        print("To install pandas: pip install pandas")
        return
    
    # Load ratings
    valence_path = os.path.join(ratings_dir, 'Valence.csv')
    arousal_path = os.path.join(ratings_dir, 'Arousal.csv')
    
    if not os.path.exists(valence_path) or not os.path.exists(arousal_path):
        print("Warning: Ratings files not found")
        return
    
    try:
        # Read CSV files
        valence_df = pd.read_csv(valence_path, header=None, names=['FileName', 'Valence'])
        arousal_df = pd.read_csv(arousal_path, header=None, names=['FileName', 'Arousal'])
    except Exception as e:
        print(f"Warning: Could not read CSV files: {e}")
        return
    
    # Get unique filenames from ratings
    valence_files = set(valence_df['FileName'])
    arousal_files = set(arousal_df['FileName'])
    rated_files = valence_files.intersection(arousal_files)
    
    # Get audio files
    audio_files = set()
    if os.path.exists(audio_dir):
        for ext in ['*.wav', '*.mp3', '*.flac']:
            audio_files.update([os.path.basename(f) for f in glob.glob(os.path.join(audio_dir, ext))])
    
    print(f"\n=== Ratings Coverage ===")
    print(f"Files with valence ratings: {len(valence_files)}")
    print(f"Files with arousal ratings: {len(arousal_files)}")
    print(f"Files with both ratings: {len(rated_files)}")
    print(f"Audio files found: {len(audio_files)}")
    
    if audio_files:
        matched_files = rated_files.intersection(audio_files)
        print(f"Audio files with ratings: {len(matched_files)}")
        print(f"Coverage: {len(matched_files)/len(audio_files)*100:.1f}%")
        
        # Show some examples of unmatched files
        unmatched_audio = audio_files - rated_files
        unmatched_ratings = rated_files - audio_files
        
        if unmatched_audio:
            print(f"\nExample audio files without ratings:")
            for f in list(unmatched_audio)[:5]:
                print(f"  {f}")
        
        if unmatched_ratings:
            print(f"\nExample rating entries without audio files:")
            for f in list(unmatched_ratings)[:5]:
                print(f"  {f}")

def main():
    parser = argparse.ArgumentParser(description='Flatten Emo-Soundscapes audio directory structure')
    parser.add_argument('--source_base', type=str, required=True,
                        help='Base directory containing 600_Sounds and 613_MixedSounds')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for flattened audio files')
    parser.add_argument('--ratings_dir', type=str,
                        help='Directory containing Valence.csv and Arousal.csv for coverage check')
    parser.add_argument('--copy', action='store_true',
                        help='Actually copy files (default is dry run)')
    parser.add_argument('--include_600', action='store_true', default=True,
                        help='Include 600_Sounds directory')
    parser.add_argument('--include_613', action='store_true', default=True,
                        help='Include 613_MixedSounds directory')
    
    args = parser.parse_args()
    
    # Build source directories list
    source_dirs = []
    if args.include_600:
        source_dirs.append(os.path.join(args.source_base, '600_Sounds'))
    if args.include_613:
        source_dirs.append(os.path.join(args.source_base, '613_MixedSounds'))
    
    if not source_dirs:
        print("Error: No source directories specified")
        return
    
    print("Emo-Soundscapes Audio Directory Flattener")
    print("=" * 50)
    print(f"Source directories: {source_dirs}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mode: {'COPY FILES' if args.copy else 'DRY RUN'}")
    
    # Flatten directory structure
    processed_files = flatten_audio_directory(source_dirs, args.output_dir, args.copy)
    
    # Check ratings coverage if requested
    if args.ratings_dir and args.copy:
        check_ratings_coverage(args.output_dir, args.ratings_dir)

if __name__ == '__main__':
    main() 