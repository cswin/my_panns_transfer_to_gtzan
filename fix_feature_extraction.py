#!/usr/bin/env python3

import os
import sys
import librosa
import numpy as np
from collections import defaultdict
import soundfile as sf

def diagnose_failed_files():
    """Diagnose which files are failing to load and why."""
    
    dataset_dir = "/DATA/pliu/EmotionData/GTZAN/genres_original/"
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    print("=== GTZAN Audio File Loading Diagnosis ===")
    
    successful_files = []
    failed_files = []
    error_types = defaultdict(list)
    
    total_files = 0
    
    for genre in genres:
        genre_dir = os.path.join(dataset_dir, genre)
        print(f"\n📁 Testing {genre} files...")
        
        genre_files = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
        genre_success = 0
        genre_failed = 0
        
        for filename in sorted(genre_files):
            file_path = os.path.join(genre_dir, filename)
            total_files += 1
            
            # Test with librosa.load (current method)
            try:
                audio, sr = librosa.load(file_path, sr=32000, mono=True)
                if len(audio) > 0:
                    successful_files.append((filename, file_path))
                    genre_success += 1
                else:
                    failed_files.append((filename, "Empty audio"))
                    error_types["Empty audio"].append(filename)
                    genre_failed += 1
            except Exception as e:
                error_msg = str(e)
                failed_files.append((filename, error_msg))
                error_types[error_msg].append(filename)
                genre_failed += 1
                
                # Try alternative loading methods
                alt_success = False
                
                # Try soundfile
                try:
                    audio, sr = sf.read(file_path)
                    print(f"   ⚠️  {filename}: librosa failed, but soundfile works")
                    alt_success = True
                except Exception as sf_error:
                    pass
                
                # Try librosa with different parameters
                if not alt_success:
                    try:
                        audio, sr = librosa.load(file_path, sr=None, mono=True)
                        print(f"   ⚠️  {filename}: works with sr=None")
                        alt_success = True
                    except:
                        pass
        
        print(f"   ✅ Success: {genre_success}/{len(genre_files)} files")
        print(f"   ❌ Failed: {genre_failed}/{len(genre_files)} files")
    
    print(f"\n=== Overall Results ===")
    print(f"Total files tested: {total_files}")
    print(f"Successfully loaded: {len(successful_files)}")
    print(f"Failed to load: {len(failed_files)}")
    print(f"Success rate: {len(successful_files)/total_files*100:.1f}%")
    
    print(f"\n=== Error Analysis ===")
    for error_type, files in error_types.items():
        print(f"{error_type}: {len(files)} files")
        if len(files) <= 5:
            print(f"   Files: {files}")
        else:
            print(f"   Sample files: {files[:5]} ...")
    
    # Check file properties of failed files
    if failed_files:
        print(f"\n=== Failed File Analysis ===")
        sample_failed = failed_files[:5]
        for filename, error in sample_failed:
            file_path = os.path.join(dataset_dir, filename.split('.')[0], filename)
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                print(f"{filename}:")
                print(f"   Size: {stat.st_size} bytes")
                print(f"   Error: {error}")
                
                # Try to get audio info without loading
                try:
                    info = sf.info(file_path)
                    print(f"   Format: {info.format}, Subtype: {info.subtype}")
                    print(f"   Sample rate: {info.samplerate}, Channels: {info.channels}")
                    print(f"   Duration: {info.duration:.2f}s")
                except Exception as info_error:
                    print(f"   Info error: {info_error}")
    
    return successful_files, failed_files

def create_fixed_features():
    """Re-run feature extraction with better error handling."""
    
    print("\n" + "="*50)
    print("CREATING FIXED FEATURES WITH ENHANCED ERROR HANDLING")
    print("="*50)
    
    # This would be the enhanced version of pack_audio_files_to_hdf5
    # with better error handling and alternative loading methods
    
    dataset_dir = "/DATA/pliu/EmotionData/GTZAN/genres_original/"
    workspace = "/home/pengliu/Private/panns_transfer_to_gtzan/"
    
    print("Recommendations for fixing feature extraction:")
    print("1. Add try-catch with alternative loading methods")
    print("2. Log all failed files for manual inspection")
    print("3. Use soundfile as backup for librosa failures")
    print("4. Handle different sample rates gracefully")
    print("5. Skip corrupted files but log them")
    
    return

if __name__ == '__main__':
    # Run the diagnosis
    successful, failed = diagnose_failed_files()
    
    if len(failed) > 0:
        print(f"\n🚨 CRITICAL: {len(failed)} files are failing to load!")
        print("This explains why only 325/1000 files are processed.")
        print("\nRecommended next steps:")
        print("1. Check if files are corrupted")
        print("2. Try alternative audio loading libraries")
        print("3. Convert problematic files to standard format")
        print("4. Update feature extraction with better error handling")
    else:
        print("\n✅ All files load successfully - issue must be elsewhere") 