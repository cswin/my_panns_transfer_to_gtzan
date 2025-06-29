#!/usr/bin/env python3
"""
Random WAV File Splitter

This script randomly selects 50% of WAV files from a source folder and moves them to a destination folder.
Useful for creating train/validation splits or other data partitioning tasks.
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List


def get_wav_files(folder_path: str) -> List[str]:
    """
    Get all WAV files from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing WAV files
        
    Returns:
        List[str]: List of full paths to WAV files
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Source folder does not exist: {folder_path}")
    
    wav_files = list(folder.glob("*.wav"))
    return [str(f) for f in wav_files]


def create_destination_folder(dest_path: str) -> None:
    """
    Create destination folder if it doesn't exist.
    
    Args:
        dest_path (str): Path to the destination folder
    """
    dest_folder = Path(dest_path)
    dest_folder.mkdir(parents=True, exist_ok=True)


def move_random_files(source_files: List[str], dest_folder: str, percentage: float = 0.5) -> List[str]:
    """
    Randomly select and move files to destination folder.
    
    Args:
        source_files (List[str]): List of source file paths
        dest_folder (str): Destination folder path
        percentage (float): Percentage of files to move (default: 0.5 for 50%)
        
    Returns:
        List[str]: List of moved file paths
    """
    if not source_files:
        print("No WAV files found in source folder.")
        return []
    
    # Calculate number of files to move
    num_files_to_move = int(len(source_files) * percentage)
    
    # Randomly select files
    files_to_move = random.sample(source_files, num_files_to_move)
    
    # Move files
    moved_files = []
    for file_path in files_to_move:
        file_name = Path(file_path).name
        dest_path = Path(dest_folder) / file_name
        
        try:
            shutil.move(file_path, dest_path)
            moved_files.append(str(dest_path))
            print(f"Moved: {file_name}")
        except Exception as e:
            print(f"Error moving {file_name}: {e}")
    
    return moved_files


def main():
    parser = argparse.ArgumentParser(
        description="Randomly select and move 50% of WAV files from source to destination folder"
    )
    parser.add_argument(
        "source_folder",
        help="Path to the source folder containing WAV files"
    )
    parser.add_argument(
        "destination_folder",
        help="Path to the destination folder where files will be moved"
    )
    parser.add_argument(
        "--percentage",
        type=float,
        default=0.5,
        help="Percentage of files to move (default: 0.5 for 50%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible results"
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    try:
        # Get WAV files from source folder
        print(f"Scanning for WAV files in: {args.source_folder}")
        wav_files = get_wav_files(args.source_folder)
        print(f"Found {len(wav_files)} WAV files")
        
        if not wav_files:
            print("No WAV files found. Exiting.")
            return
        
        # Create destination folder
        print(f"Creating destination folder: {args.destination_folder}")
        create_destination_folder(args.destination_folder)
        
        # Move random files
        print(f"Moving {args.percentage * 100}% of files ({int(len(wav_files) * args.percentage)} files)...")
        moved_files = move_random_files(wav_files, args.destination_folder, args.percentage)
        
        print(f"\nOperation completed successfully!")
        print(f"Files moved: {len(moved_files)}")
        print(f"Files remaining in source: {len(wav_files) - len(moved_files)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 