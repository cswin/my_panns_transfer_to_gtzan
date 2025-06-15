#!/usr/bin/env python3
"""
Extract mel-spectrogram features from Emo-Soundscapes dataset.

This script:
1. Loads audio files from the Emo-Soundscapes dataset
2. Extracts mel-spectrogram features 
3. Loads valence and arousal ratings
4. Saves everything to HDF5 format for training
"""

import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))

import numpy as np
import pandas as pd
import librosa
import h5py
import argparse
from tqdm import tqdm
import glob

from utilities import create_folder
from config import sample_rate, clip_samples, mel_bins, fmin, fmax, window_size, hop_size


def extract_melspectrogram_segments(audio_path, sr=32000, clip_duration=6.0, segment_duration=1.0):
    """Extract mel-spectrogram segments from audio file (similar to GTZAN approach).
    
    Args:
        audio_path: str, path to audio file
        sr: int, sample rate
        clip_duration: float, total duration of clip in seconds
        segment_duration: float, duration of each segment in seconds
        
    Returns:
        mel_specs: list of np.array, mel-spectrograms for each segment
    """
    try:
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr, duration=clip_duration)
        
        # Pad or truncate to exact duration
        target_length = int(sr * clip_duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # Extract segments (similar to GTZAN: 1-second segments)
        segment_samples = int(sr * segment_duration)
        num_segments = int(clip_duration / segment_duration)
        
        mel_specs = []
        for i in range(num_segments):
            start_idx = i * segment_samples
            end_idx = start_idx + segment_samples
            segment = audio[start_idx:end_idx]
            
            # Extract mel-spectrogram for this segment
            mel_spec = librosa.feature.melspectrogram(
                y=segment,
                sr=sr,
                n_fft=window_size,
                hop_length=hop_size,
                n_mels=mel_bins,
                fmin=fmin,
                fmax=fmax
            )
            
            # Convert to log scale
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Transpose to (time_steps, mel_bins)
            mel_spec = mel_spec.T
            
            mel_specs.append(mel_spec)
        
        return mel_specs
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def extract_melspectrogram(audio_path, sr=32000, duration=1.0):
    """Extract mel-spectrogram from audio file (single segment).
    
    Args:
        audio_path: str, path to audio file
        sr: int, sample rate
        duration: float, duration in seconds
        
    Returns:
        mel_spec: np.array, mel-spectrogram (time_steps, mel_bins)
    """
    try:
        # Load audio
        audio, _ = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Pad or truncate to exact duration
        target_length = int(sr * duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=window_size,
            hop_length=hop_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax
        )
        
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to (time_steps, mel_bins)
        mel_spec = mel_spec.T
        
        return mel_spec
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None


def load_emotion_ratings(ratings_dir):
    """Load valence and arousal ratings from CSV files.
    
    Args:
        ratings_dir: str, directory containing Arousal.csv and Valence.csv
        
    Returns:
        ratings_dict: dict, {filename: {'valence': float, 'arousal': float}}
    """
    valence_path = os.path.join(ratings_dir, 'Valence.csv')
    arousal_path = os.path.join(ratings_dir, 'Arousal.csv')
    
    # Load ratings with proper CSV parsing
    try:
        valence_df = pd.read_csv(valence_path, header=None, names=['FileName', 'Rating'])
        arousal_df = pd.read_csv(arousal_path, header=None, names=['FileName', 'Rating'])
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        print("Trying alternative parsing...")
        
        # Try reading as raw text and parsing manually (in case CSV is malformed)
        ratings_dict = {}
        
        # Parse valence file
        try:
            with open(valence_path, 'r') as f:
                content = f.read().strip()
                # Split by filename pattern (assuming .wav appears in filenames)
                entries = content.split('.wav,')
                for i, entry in enumerate(entries):
                    if i == len(entries) - 1:  # Last entry
                        if '.wav' in entry:
                            parts = entry.rsplit(',', 1)
                            if len(parts) == 2:
                                filename = parts[0] + '.wav'
                                rating = float(parts[1])
                                ratings_dict[filename] = {'valence': rating}
                    else:
                        parts = entry.rsplit(',', 1)
                        if len(parts) == 2:
                            filename = parts[0].split()[-1] + '.wav'  # Get last word + .wav
                            rating = float(parts[1])
                            if filename not in ratings_dict:
                                ratings_dict[filename] = {}
                            ratings_dict[filename]['valence'] = rating
        except Exception as e2:
            print(f"Manual valence parsing failed: {e2}")
            return {}
        
        # Parse arousal file
        try:
            with open(arousal_path, 'r') as f:
                content = f.read().strip()
                entries = content.split('.wav,')
                for i, entry in enumerate(entries):
                    if i == len(entries) - 1:  # Last entry
                        if '.wav' in entry:
                            parts = entry.rsplit(',', 1)
                            if len(parts) == 2:
                                filename = parts[0] + '.wav'
                                rating = float(parts[1])
                                if filename in ratings_dict:
                                    ratings_dict[filename]['arousal'] = rating
                    else:
                        parts = entry.rsplit(',', 1)
                        if len(parts) == 2:
                            filename = parts[0].split()[-1] + '.wav'
                            rating = float(parts[1])
                            if filename in ratings_dict:
                                ratings_dict[filename]['arousal'] = rating
        except Exception as e3:
            print(f"Manual arousal parsing failed: {e3}")
            return {}
        
        # Filter out entries that don't have both ratings
        complete_ratings = {k: v for k, v in ratings_dict.items() 
                           if 'valence' in v and 'arousal' in v}
        
        print(f"Manual parsing: Loaded ratings for {len(complete_ratings)} files")
        return complete_ratings
    
    # Standard CSV parsing worked
    ratings_dict = {}
    
    # Merge valence and arousal ratings
    for _, row in valence_df.iterrows():
        filename = row['FileName']
        ratings_dict[filename] = {'valence': row['Rating']}
    
    for _, row in arousal_df.iterrows():
        filename = row['FileName']
        if filename in ratings_dict:
            ratings_dict[filename]['arousal'] = row['Rating']
        else:
            ratings_dict[filename] = {'arousal': row['Rating']}
    
    # Filter out entries that don't have both ratings
    complete_ratings = {k: v for k, v in ratings_dict.items() 
                       if 'valence' in v and 'arousal' in v}
    
    print(f"Loaded ratings for {len(complete_ratings)} files")
    return complete_ratings


def find_audio_files_recursive(audio_dirs):
    """Find all audio files in multiple directories, including subdirectories.
    
    Args:
        audio_dirs: list of directories to search
        
    Returns:
        dict: {filename: full_path} mapping
    """
    audio_files = {}
    audio_patterns = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.aac']
    
    for audio_dir in audio_dirs:
        if not os.path.exists(audio_dir):
            print(f"Warning: Directory does not exist: {audio_dir}")
            continue
            
        print(f"Searching in: {audio_dir}")
        
        for pattern in audio_patterns:
            # Search recursively
            pattern_path = os.path.join(audio_dir, '**', pattern)
            files = glob.glob(pattern_path, recursive=True)
            
            for file_path in files:
                filename = os.path.basename(file_path)
                if filename in audio_files:
                    print(f"Warning: Duplicate filename found: {filename}")
                audio_files[filename] = file_path
        
        print(f"Found {len([f for f in audio_files.values() if f.startswith(audio_dir)])} files in {audio_dir}")
    
    return audio_files


def extract_features(args):
    """Main feature extraction function."""
    
    # Parse audio directories - can be multiple directories
    if isinstance(args.audio_dir, str):
        audio_dirs = [args.audio_dir]
    else:
        audio_dirs = args.audio_dir
    
    ratings_dir = args.ratings_dir
    output_dir = args.output_dir
    
    create_folder(output_dir)
    
    # Load emotion ratings
    print("Loading emotion ratings...")
    ratings_dict = load_emotion_ratings(ratings_dir)
    
    if not ratings_dict:
        print("Error: No ratings loaded!")
        return
    
    # Find all audio files recursively
    print("Finding audio files...")
    audio_files_dict = find_audio_files_recursive(audio_dirs)
    print(f"Found {len(audio_files_dict)} total audio files")
    
    # Match audio files with ratings
    rated_files = []
    for filename in ratings_dict.keys():
        # Try exact match first
        if filename in audio_files_dict:
            rated_files.append((filename, audio_files_dict[filename]))
        else:
            # Try without extension and re-add .wav
            base_name = os.path.splitext(filename)[0] + '.wav'
            if base_name in audio_files_dict:
                rated_files.append((filename, audio_files_dict[base_name]))
            else:
                # Try case-insensitive match
                for audio_filename, audio_path in audio_files_dict.items():
                    if audio_filename.lower() == filename.lower():
                        rated_files.append((filename, audio_path))
                        break
    
    print(f"Found {len(rated_files)} audio files with emotion ratings")
    
    if len(rated_files) == 0:
        print("No audio files found with matching ratings!")
        print("Sample rating filenames:", list(ratings_dict.keys())[:5])
        print("Sample audio filenames:", list(audio_files_dict.keys())[:5])
        return
    
    # Extract features
    features = []
    valence_ratings = []
    arousal_ratings = []
    audio_names = []
    
    print("Extracting features...")
    for rating_filename, audio_file_path in tqdm(rated_files):
        # Extract mel-spectrogram segments (6 segments of 1 second each)
        mel_specs = extract_melspectrogram_segments(audio_file_path)
        
        if mel_specs is not None:
            # Add each segment as a separate sample (similar to GTZAN approach)
            for i, mel_spec in enumerate(mel_specs):
                features.append(mel_spec)
                valence_ratings.append(ratings_dict[rating_filename]['valence'])
                arousal_ratings.append(ratings_dict[rating_filename]['arousal'])
                # Add segment index to filename for tracking
                segment_name = f"{rating_filename}_seg{i}"
                audio_names.append(segment_name)
    
    print(f"Successfully extracted features from {len(features)} files")
    
    if len(features) == 0:
        print("No features extracted!")
        return
    
    # Convert to numpy arrays
    features = np.array(features)
    valence_ratings = np.array(valence_ratings)
    arousal_ratings = np.array(arousal_ratings)
    audio_names = np.array([name.encode() for name in audio_names])
    
    # Save to HDF5
    output_path = os.path.join(output_dir, 'emotion_features.h5')
    print(f"Saving features to {output_path}")
    
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset('feature', data=features, dtype=np.float32)
        hf.create_dataset('valence', data=valence_ratings, dtype=np.float32)
        hf.create_dataset('arousal', data=arousal_ratings, dtype=np.float32)
        hf.create_dataset('audio_name', data=audio_names)
    
    print(f"Feature extraction complete!")
    print(f"Shape of features: {features.shape}")
    print(f"Valence range: [{valence_ratings.min():.3f}, {valence_ratings.max():.3f}]")
    print(f"Arousal range: [{arousal_ratings.min():.3f}, {arousal_ratings.max():.3f}]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features from Emo-Soundscapes dataset')
    parser.add_argument('--audio_dir', type=str, required=True, nargs='+',
                        help='Directory(ies) containing audio files (can specify multiple)')
    parser.add_argument('--ratings_dir', type=str, required=True,
                        help='Directory containing Emo-Soundscapes-Ratings (Valence.csv and Arousal.csv)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for features')
    
    args = parser.parse_args()
    
    # Validate inputs
    for audio_dir in args.audio_dir:
        if not os.path.exists(audio_dir):
            raise ValueError(f"Audio directory not found: {audio_dir}")
    if not os.path.exists(args.ratings_dir):
        raise ValueError(f"Ratings directory not found: {args.ratings_dir}")
    
    extract_features(args) 