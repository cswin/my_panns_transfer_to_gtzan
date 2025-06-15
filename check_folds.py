import h5py
import numpy as np

with h5py.File('/home/pengliu/Private/panns_transfer_to_gtzan/features/minidata_features.h5', 'r') as hf:
    print('Total segments:', len(hf['feature']))
    folds = np.unique(hf['fold'][:])
    print('Available folds:', folds)
    print('Segments per fold:')
    for fold in folds:
        count = np.sum(hf['fold'][:] == fold)
        print(f'  Fold {fold}: {count} segments')
    
    # Check current train/val split
    print('\nCurrent split (validation_folds=[1, 6, 10]):')
    val_folds = [1, 6, 10]
    train_mask = ~np.isin(hf['fold'][:], val_folds)
    val_mask = np.isin(hf['fold'][:], val_folds)
    print(f'  Training segments: {np.sum(train_mask)}')
    print(f'  Validation segments: {np.sum(val_mask)}')
    
    # Check feature shapes
    print('\nFeature shapes:')
    print('Overall feature shape:', hf['feature'].shape)
    print('First 10 feature shapes:')
    for i in range(min(10, len(hf['feature']))):
        feature = hf['feature'][i]
        print(f'  Segment {i}: {feature.shape}')
    
    # Check unique shapes
    shapes = []
    for i in range(len(hf['feature'])):
        shapes.append(hf['feature'][i].shape)
    unique_shapes = list(set(shapes))
    print(f'\nUnique feature shapes: {unique_shapes}')
    print(f'Number of unique shapes: {len(unique_shapes)}')
    
    # Check how segments are organized
    print('\nSegment organization:')
    print('First 20 entries:')
    print('Audio names:', [name.decode() for name in hf['audio_name'][:20]])
    print('Segment indices:', hf['segment_idx'][:20])
    print('Folds:', hf['fold'][:20])
    
    # Check unique audio files
    unique_audio_names = np.unique([name.decode() for name in hf['audio_name'][:]])
    print(f'\nUnique audio files: {len(unique_audio_names)}')
    print('Audio files:', unique_audio_names) 