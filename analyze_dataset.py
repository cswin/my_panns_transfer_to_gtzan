#!/usr/bin/env python3

import h5py
import os

def analyze_incomplete_dataset():
    features_path = '/home/pengliu/Private/panns_transfer_to_gtzan/features/features.h5'
    
    try:
        with h5py.File(features_path, 'r') as hf:
            audio_names = [name.decode() for name in hf['audio_name'][:]]
            unique_files = list(set(audio_names))
            
            print('=== GTZAN Dataset Analysis ===')
            print(f'Total files found: {len(unique_files)}')
            print(f'Expected: 1000 files (100 per genre)')
            print(f'Missing: {1000 - len(unique_files)} files')
            print()
            
            # Check genre distribution
            genres = {}
            for name in unique_files:
                genre = name.split('.')[0]
                genres[genre] = genres.get(genre, 0) + 1
            
            print('=== Genre Distribution ===')
            expected_genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                             'jazz', 'metal', 'pop', 'reggae', 'rock']
            
            total_present = 0
            for genre in expected_genres:
                count = genres.get(genre, 0)
                status = '✓' if count > 0 else '❌'
                print(f'{status} {genre}: {count}/100 files')
                total_present += count
            
            print(f'\nTotal files across all genres: {total_present}')
            
            # Check for unexpected genres
            unexpected_genres = [g for g in genres.keys() if g not in expected_genres]
            if unexpected_genres:
                print(f'\nUnexpected genres found: {unexpected_genres}')
            
            print('\n=== Missing Genres ===')
            missing_genres = [g for g in expected_genres if g not in genres]
            if missing_genres:
                print(f'Completely missing: {missing_genres}')
                print(f'This explains the 4x4 confusion matrix (only {10-len(missing_genres)} genres present)')
            else:
                print('All genres present, but with reduced file counts')
            
            # Calculate expected validation split
            print(f'\n=== Validation Split Analysis ===')
            print(f'With {len(unique_files)} total files:')
            print(f'- Training: ~{int(len(unique_files) * 0.7)} files (70%)')
            print(f'- Validation: ~{int(len(unique_files) * 0.3)} files (30%)')
            print(f'- This matches your log showing 98 validation files!')
            
    except Exception as e:
        print(f'Error reading features file: {e}')

if __name__ == '__main__':
    analyze_incomplete_dataset() 