import os
import sys
import numpy as np
import argparse
import h5py
import librosa
import matplotlib.pyplot as plt
import time
import csv
import math
import re
import random
import torch
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

import config
from utilities import create_folder, traverse_folder, float32_to_int16


def to_one_hot(k, classes_num):
    target = np.zeros(classes_num)
    target[k] = 1
    return target


def pad_truncate_sequence(x, max_len):
    if len(x) < max_len:
        return np.concatenate((x, np.zeros(max_len - len(x))))
    else:
        return x[0 : max_len]


def segment_audio(audio, segment_length):
    """
    Segment audio into sequential, non-overlapping chunks.
    Similar to how humans process audio in real-time.
    """
    segments = []
    num_segments = len(audio) // segment_length
    
    # Process complete segments
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segments.append(audio[start:end])
    
    # Handle the last incomplete segment if it exists
    remaining = len(audio) % segment_length
    if remaining > 0:
        last_segment = np.pad(audio[-remaining:], 
                            (0, segment_length - remaining), 
                            mode='constant')
        segments.append(last_segment)
    
    return segments


def extract_features(audio, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
    """
    Extract log mel spectrogram features from audio segment.
    Returns features in the format expected by the model.
    """
    # Initialize feature extractors
    spectrogram_extractor = Spectrogram(n_fft=window_size, 
        hop_length=hop_size, 
        win_length=window_size, 
        window='hann', 
        center=True, 
        pad_mode='reflect', 
        freeze_parameters=True)
        
    logmel_extractor = LogmelFilterBank(sr=sample_rate, 
        n_fft=window_size, 
        n_mels=mel_bins, 
        fmin=fmin, 
        fmax=fmax, 
        ref=1.0,
        amin=1e-10, 
        top_db=None, 
        freeze_parameters=True)

    # Convert audio to torch tensor
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # (1, audio_length)
    
    # Extract features
    x = spectrogram_extractor(audio_tensor)   # (1, 1, time_steps, freq_bins)
    x = logmel_extractor(x)    # (1, 1, time_steps, mel_bins)
    
    # Remove batch and channel dimensions to get (time_steps, mel_bins)
    x = x.squeeze(0).squeeze(0)  # (time_steps, mel_bins)
    
    return x.numpy()


def create_indexes(hdf5_path, train_ratio=0.7):
    """Create train and validate indexes for the dataset.
    Split by audio files, not segments, to prevent data leakage.
    Args:
        hdf5_path: str, path to the features hdf5 file
        train_ratio: float, ratio of audio files for training (default 0.7 for 70/30 split)
    """
    with h5py.File(hdf5_path, 'r') as hf:
        total_segments = len(hf['segment_idx'])
        audio_names = [name.decode() for name in hf['audio_name'][:]]
        
    # Get unique audio files
    unique_audio_files = list(set(audio_names))
    unique_audio_files.sort()  # Sort for reproducible results
    
    # Split audio files using random seed for reproducibility
    np.random.seed(42)  # Fixed seed for reproducible splits
    num_train_files = int(len(unique_audio_files) * train_ratio)
    
    # Shuffle and split
    shuffled_files = unique_audio_files.copy()
    np.random.shuffle(shuffled_files)
    
    train_audio_files = shuffled_files[:num_train_files]
    validate_audio_files = shuffled_files[num_train_files:]
    
    # Get segment indexes for each split
    train_indexes = []
    validate_indexes = []
    
    for i in range(total_segments):
        audio_name = audio_names[i]
        if audio_name in train_audio_files:
            train_indexes.append(i)
        elif audio_name in validate_audio_files:
            validate_indexes.append(i)
    
    # Convert to numpy arrays
    train_indexes = np.array(train_indexes)
    validate_indexes = np.array(validate_indexes)
    
    print(f'Total audio files: {len(unique_audio_files)}')
    print(f'Train audio files: {len(train_audio_files)} ({len(train_audio_files)/len(unique_audio_files)*100:.1f}%)')
    print(f'Validation audio files: {len(validate_audio_files)} ({len(validate_audio_files)/len(unique_audio_files)*100:.1f}%)')
    print(f'Train segments: {len(train_indexes)}')
    print(f'Validation segments: {len(validate_indexes)}')
    print(f'Train audio files: {sorted(train_audio_files)}')
    print(f'Validation audio files: {sorted(validate_audio_files)}')
    
    # Create index files
    train_indexes_path = hdf5_path[:-3] + '_train.h5'
    validate_indexes_path = hdf5_path[:-3] + '_validate.h5'
    
    with h5py.File(train_indexes_path, 'w') as hf:
        hf.create_dataset('audio_indexes', data=train_indexes)
        hf.create_dataset('segment_indexes', data=train_indexes)
        
    with h5py.File(validate_indexes_path, 'w') as hf:
        hf.create_dataset('audio_indexes', data=validate_indexes)
        hf.create_dataset('segment_indexes', data=validate_indexes)
        
    print(f'Created index files:\n{train_indexes_path}\n{validate_indexes_path}')


def pack_audio_files_to_hdf5(args):
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    mini_data = args.mini_data

    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx

    # Segment parameters - 1 second non-overlapping segments
    segment_length = sample_rate  # 1 second segments

    # Paths
    audios_dir = os.path.join(dataset_dir)

    if mini_data:
        packed_hdf5_path = os.path.join(workspace, 'features', 'minidata_features.h5')
    else:
        packed_hdf5_path = os.path.join(workspace, 'features', 'features.h5')
    create_folder(os.path.dirname(packed_hdf5_path))

    (audio_names, audio_paths) = traverse_folder(audios_dir)
    
    # Filter out non-audio files and empty names
    valid_audio_names = []
    valid_audio_paths = []
    for name, path in zip(audio_names, audio_paths):
        if name and name.split('.')[0] and name.split('.')[0] in lb_to_idx:
            valid_audio_names.append(name)
            valid_audio_paths.append(path)
    
    audio_names = sorted(valid_audio_names)
    audio_paths = sorted(valid_audio_paths)

    meta_dict = {
        'audio_name': np.array(audio_names), 
        'audio_path': np.array(audio_paths), 
        'target': np.array([lb_to_idx[audio_name.split('.')[0]] for audio_name in audio_names]), 
        'fold': np.arange(len(audio_names)) % 10 + 1}
    
    if mini_data:
        mini_num = 10
        total_num = len(meta_dict['audio_name'])
        random_state = np.random.RandomState(1234)
        indexes = random_state.choice(total_num, size=mini_num, replace=False)
        for key in meta_dict.keys():
            meta_dict[key] = meta_dict[key][indexes]

    audios_num = len(meta_dict['audio_name'])

    feature_time = time.time()
    
    # First pass to count total segments
    total_segments = 0
    valid_files = []
    failed_files = []
    
    print(f"Loading and validating {audios_num} audio files...")
    
    for n in range(audios_num):
        audio_path = meta_dict['audio_path'][n]
        audio_name = meta_dict['audio_name'][n]
        
        # Try multiple loading methods
        audio_loaded = False
        audio = None
        
        # Method 1: librosa with target sample rate
        try:
            (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
            if len(audio) > 0:
                audio_loaded = True
        except Exception as e:
            print(f"Method 1 failed for {audio_name}: {e}")
        
        # Method 2: librosa with original sample rate, then resample
        if not audio_loaded:
            try:
                (audio, fs) = librosa.core.load(audio_path, sr=None, mono=True)
                if fs != sample_rate:
                    audio = librosa.resample(audio, orig_sr=fs, target_sr=sample_rate)
                if len(audio) > 0:
                    audio_loaded = True
                    print(f"Method 2 success for {audio_name} (resampled from {fs}Hz)")
            except Exception as e:
                print(f"Method 2 failed for {audio_name}: {e}")
        
        # Method 3: soundfile then resample
        if not audio_loaded:
            try:
                import soundfile as sf
                audio, fs = sf.read(audio_path)
                if len(audio.shape) > 1:  # Convert stereo to mono
                    audio = np.mean(audio, axis=1)
                if fs != sample_rate:
                    audio = librosa.resample(audio, orig_sr=fs, target_sr=sample_rate)
                if len(audio) > 0:
                    audio_loaded = True
                    print(f"Method 3 success for {audio_name} (soundfile + resample)")
            except Exception as e:
                print(f"Method 3 failed for {audio_name}: {e}")
        
        if audio_loaded:
            try:
                segments = segment_audio(audio, segment_length)
                total_segments += len(segments)
                valid_files.append(n)
            except Exception as e:
                print(f"Segmentation failed for {audio_name}: {e}")
                failed_files.append((audio_name, f"Segmentation error: {e}"))
        else:
            print(f"❌ FAILED TO LOAD: {audio_name}")
            failed_files.append((audio_name, "All loading methods failed"))
    
    print(f"✅ Successfully loaded: {len(valid_files)}/{audios_num} files")
    print(f"❌ Failed to load: {len(failed_files)}/{audios_num} files")
    
    if failed_files:
        print("Failed files:")
        for name, error in failed_files:
            print(f"  - {name}: {error}")
    
    if len(valid_files) < audios_num * 0.8:  # Less than 80% success rate
        print(f"⚠️  WARNING: Low success rate ({len(valid_files)}/{audios_num})")
        print("Consider checking audio file integrity or formats")
    
    print(f"Total segments from valid files: {total_segments}")

    with h5py.File(packed_hdf5_path, 'w') as hf:
        hf.create_dataset(
            name='audio_name', 
            shape=(total_segments,), 
            dtype='S80')
            
        hf.create_dataset(
            name='feature',
            shape=(total_segments, int(segment_length / hop_size) + 1, mel_bins),
            dtype=np.float32)

        hf.create_dataset(
            name='target', 
            shape=(total_segments, classes_num), 
            dtype=np.float32)

        hf.create_dataset(
            name='fold', 
            shape=(total_segments,), 
            dtype=np.int32)
            
        hf.create_dataset(
            name='segment_idx',
            shape=(total_segments,),
            dtype=np.int32)

        segment_idx = 0
        for n in valid_files:
            print(f"Processing file {n+1}/{audios_num}")
            audio_name = meta_dict['audio_name'][n]
            audio_path = meta_dict['audio_path'][n]
            
            # Use the same enhanced loading as above
            audio_loaded = False
            audio = None
            
            # Method 1: librosa with target sample rate
            try:
                (audio, fs) = librosa.core.load(audio_path, sr=sample_rate, mono=True)
                if len(audio) > 0:
                    audio_loaded = True
            except:
                pass
            
            # Method 2: librosa with original sample rate, then resample
            if not audio_loaded:
                try:
                    (audio, fs) = librosa.core.load(audio_path, sr=None, mono=True)
                    if fs != sample_rate:
                        audio = librosa.resample(audio, orig_sr=fs, target_sr=sample_rate)
                    if len(audio) > 0:
                        audio_loaded = True
                except:
                    pass
            
            # Method 3: soundfile then resample
            if not audio_loaded:
                try:
                    import soundfile as sf
                    audio, fs = sf.read(audio_path)
                    if len(audio.shape) > 1:  # Convert stereo to mono
                        audio = np.mean(audio, axis=1)
                    if fs != sample_rate:
                        audio = librosa.resample(audio, orig_sr=fs, target_sr=sample_rate)
                    if len(audio) > 0:
                        audio_loaded = True
                except:
                    pass
            
            if not audio_loaded:
                print(f"ERROR: Still failed to load {audio_name} in processing phase")
                continue

            # Segment the audio into non-overlapping chunks
            segments = segment_audio(audio, segment_length)

            # Process each segment sequentially
            for i, segment in enumerate(segments):
                # Extract features for the segment
                feature = extract_features(
                    segment, 
                    sample_rate, 
                    window_size, 
                    hop_size, 
                    mel_bins, 
                    fmin, 
                    fmax)

                hf['audio_name'][segment_idx] = audio_name.encode()
                hf['feature'][segment_idx] = feature
                hf['target'][segment_idx] = to_one_hot(meta_dict['target'][n], classes_num)
                hf['fold'][segment_idx] = meta_dict['fold'][n]
                hf['segment_idx'][segment_idx] = i  # Sequential index
                segment_idx += 1

    print('Write hdf5 to {}'.format(packed_hdf5_path))
    print('Time: {:.3f} s'.format(time.time() - feature_time))
    
    # Create train and validate indexes - use multiple folds for validation
    create_indexes(packed_hdf5_path, train_ratio=0.7)  # Use 70/30 split for validation


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Calculate feature for all audio files
    parser_pack_audio = subparsers.add_parser('pack_audio_files_to_hdf5')
    parser_pack_audio.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_pack_audio.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_pack_audio.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'pack_audio_files_to_hdf5':
        pack_audio_files_to_hdf5(args)
        
    else:
        raise Exception('Incorrect arguments!')