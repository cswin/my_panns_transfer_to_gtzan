#!/usr/bin/env python3
"""
Test script to demonstrate how different audio lengths are handled.
"""

import numpy as np
import librosa
from src.utils.config import sample_rate, mel_bins, fmin, fmax, window_size, hop_size

def extract_melspectrogram_full_audio(audio_path, sr=32000, max_duration=30.0, target_length=1000):
    """Extract mel-spectrogram from full audio file."""
    try:
        # Load audio with maximum duration limit
        audio, _ = librosa.load(audio_path, sr=sr, duration=max_duration)
        
        # Extract mel-spectrogram for the full audio
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
        
        original_length = mel_spec.shape[0]
        
        # Pad or truncate to target length
        if mel_spec.shape[0] < target_length:
            # Pad with zeros
            padding = np.zeros((target_length - mel_spec.shape[0], mel_bins))
            mel_spec = np.vstack([mel_spec, padding])
            action = "padded"
        elif mel_spec.shape[0] > target_length:
            # Truncate
            mel_spec = mel_spec[:target_length, :]
            action = "truncated"
        else:
            action = "exact"
        
        return mel_spec, original_length, action
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None, None

def test_audio_length_handling():
    """Test how different audio lengths are handled."""
    
    print("Audio Length Handling Test")
    print("=" * 50)
    
    # Test with a few audio files to see different lengths
    test_files = [
        "/DATA/pliu/EmotionData/Emo-Soundscapes/audio_flat/01-01-01-01-01-01-01.wav",
        "/DATA/pliu/EmotionData/Emo-Soundscapes/audio_flat/01-01-01-01-01-01-02.wav",
        "/DATA/pliu/EmotionData/Emo-Soundscapes/audio_flat/01-01-01-01-01-01-03.wav"
    ]
    
    print(f"Target length: 1000 time steps = {1000 * 320 / 32000:.1f} seconds")
    print(f"Max duration: 30.0 seconds")
    print()
    
    for i, audio_path in enumerate(test_files):
        try:
            mel_spec, original_length, action = extract_melspectrogram_full_audio(audio_path)
            
            if mel_spec is not None:
                original_duration = original_length * 320 / 32000  # seconds
                final_shape = mel_spec.shape
                
                print(f"File {i+1}: {audio_path.split('/')[-1]}")
                print(f"  Original: {original_length} time steps ({original_duration:.1f}s)")
                print(f"  Action: {action}")
                print(f"  Final shape: {final_shape}")
                print(f"  Non-zero time steps: {np.sum(np.any(mel_spec != 0, axis=1))}")
                print()
            
        except Exception as e:
            print(f"Error with file {i+1}: {e}")
            print()

if __name__ == "__main__":
    test_audio_length_handling() 