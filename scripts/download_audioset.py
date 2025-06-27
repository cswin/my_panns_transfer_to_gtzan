#!/usr/bin/env python3
"""
Download AudioSet audio files using yt-dlp.
This script replaces the missing utils/dataset.py functionality.
"""

import os
import sys
import pandas as pd
import subprocess
import argparse
import logging
from tqdm import tqdm
import time

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('download_audioset.log'),
            logging.StreamHandler()
        ]
    )

def download_audio_segment(youtube_id, start_time, end_time, output_path):
    """Download a specific audio segment from YouTube using yt-dlp."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # yt-dlp command to download audio segment
        cmd = [
            'yt-dlp',
            f'https://www.youtube.com/watch?v={youtube_id}',
            '--extract-audio',
            '--audio-format', 'wav',
            '--audio-quality', '0',
            '--postprocessor-args', f'ffmpeg:-ss {start_time} -t {end_time - start_time}',
            '-o', output_path,
            '--quiet'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            return True
        else:
            logging.warning(f"Failed to download {youtube_id}: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logging.warning(f"Timeout downloading {youtube_id}")
        return False
    except Exception as e:
        logging.error(f"Error downloading {youtube_id}: {e}")
        return False

def download_wavs_from_csv(csv_path, audios_dir, max_files=None):
    """Download audio files from a CSV file containing YouTube segments."""
    # Read CSV file with explicit column names for AudioSet format
    df = pd.read_csv(csv_path, comment='#', names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'], skipinitialspace=True)
    
    # Print column names for debugging
    logging.info(f"CSV columns: {list(df.columns)}")
    logging.info(f"First few rows: {df.head()}")
    
    # Filter out rows without YouTube IDs
    df = df[df['YTID'].notna()]
    
    if max_files:
        df = df.head(max_files)
    
    logging.info(f"Downloading {len(df)} audio segments to {audios_dir}")
    
    successful_downloads = 0
    failed_downloads = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading"):
        youtube_id = row['YTID']
        start_time = row['start_seconds']
        end_time = row['end_seconds']
        
        # Create output filename
        output_filename = f"{youtube_id}_{start_time}_{end_time}.wav"
        output_path = os.path.join(audios_dir, output_filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            logging.info(f"File already exists: {output_filename}")
            successful_downloads += 1
            continue
        
        # Download the audio segment
        if download_audio_segment(youtube_id, start_time, end_time, output_path):
            successful_downloads += 1
        else:
            failed_downloads += 1
        
        # Small delay to be respectful to YouTube
        time.sleep(0.5)
    
    logging.info(f"Download completed: {successful_downloads} successful, {failed_downloads} failed")
    return successful_downloads, failed_downloads

def main():
    parser = argparse.ArgumentParser(description='Download AudioSet audio files using yt-dlp')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to CSV file with audio segments')
    parser.add_argument('--audios_dir', type=str, required=True,
                        help='Directory to save audio files')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to download (for testing)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Check if yt-dlp is installed
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("yt-dlp is not installed. Please install it with: pip install yt-dlp")
        sys.exit(1)
    
    # Download audio files
    download_wavs_from_csv(args.csv_path, args.audios_dir, args.max_files)

if __name__ == '__main__':
    main() 