import numpy as np
import h5py
import torch
import pandas as pd
import os


def collate_fn(list_data_dict):
    """
    Collate input features and targets for audio-level processing.
    Args:
        list_data_dict: list of dict, each dict contains:
            'feature': (time_steps, mel_bins)
            'target': (classes_num,)
            'audio_name': str
    Returns:
        np_data_dict: dict containing:
            'feature': (batch_size, time_steps, mel_bins)
            'target': (batch_size, classes_num)
            'audio_name': list of str
    """
    np_data_dict = {}
    
    # Stack features
    features = np.array([data_dict['feature'] for data_dict in list_data_dict])
    np_data_dict['feature'] = torch.Tensor(features)
    
    # Stack targets
    targets = np.array([data_dict['target'] for data_dict in list_data_dict])
    np_data_dict['target'] = torch.Tensor(targets)
    
    # Collect audio names
    audio_names = [data_dict['audio_name'] for data_dict in list_data_dict]
    np_data_dict['audio_name'] = audio_names
    
    return np_data_dict


def emotion_collate_fn(list_data_dict):
    """
    Collate input features and targets for emotion regression.
    Args:
        list_data_dict: list of dict, each dict contains:
            'feature': (time_steps, mel_bins)
            'valence': float
            'arousal': float
            'audio_name': str
    Returns:
        np_data_dict: dict containing:
            'feature': (batch_size, time_steps, mel_bins)
            'valence': (batch_size,)
            'arousal': (batch_size,)
            'audio_name': list of str
    """
    np_data_dict = {}
    
    # Stack features
    features = np.array([data_dict['feature'] for data_dict in list_data_dict])
    np_data_dict['feature'] = torch.Tensor(features)
    
    # Stack valence and arousal targets
    valence_targets = np.array([data_dict['valence'] for data_dict in list_data_dict])
    arousal_targets = np.array([data_dict['arousal'] for data_dict in list_data_dict])
    np_data_dict['valence'] = torch.Tensor(valence_targets)
    np_data_dict['arousal'] = torch.Tensor(arousal_targets)
    
    # Collect audio names
    audio_names = [data_dict['audio_name'] for data_dict in list_data_dict]
    np_data_dict['audio_name'] = audio_names
    
    return np_data_dict


class GtzanDataset(object):
    def __init__(self):
        """This class takes the meta of an audio clip as input, and return 
        the feature and target of the audio clip. This class is used by DataLoader.
        """
        pass
        
    def __getitem__(self, meta):
        """Load feature and target of an audio clip.
        Args:
            meta: {
                'hdf5_path': str, 
                'index_in_hdf5': int}
        Returns:
            data_dict: {
                'feature': (time_steps, mel_bins)
                'target': (classes_num,),
                'audio_name': str
            }
        """
        hdf5_path = meta['hdf5_path']
        index_in_hdf5 = meta['index_in_hdf5']
        
        with h5py.File(hdf5_path, 'r') as hf:
            feature = hf['feature'][index_in_hdf5]
            target = hf['target'][index_in_hdf5]
            audio_name = hf['audio_name'][index_in_hdf5].decode()
            
        data_dict = {
            'feature': feature,
            'target': target,
            'audio_name': audio_name}
            
        return data_dict


class EmoSoundscapesDataset(object):
    def __init__(self):
        """Dataset class for Emo-Soundscapes emotion regression."""
        pass
        
    def __getitem__(self, meta):
        """Load feature and emotion targets for a single audio clip.
        Args:
            meta: {
                'hdf5_path': str, 
                'index_in_hdf5': int}
        Returns:
            data_dict: {
                'feature': (time_steps, mel_bins)
                'valence': float,
                'arousal': float,
                'audio_name': str
            }
        """
        hdf5_path = meta['hdf5_path']
        index_in_hdf5 = meta['index_in_hdf5']
        
        with h5py.File(hdf5_path, 'r') as hf:
            feature = hf['feature'][index_in_hdf5]
            valence = float(hf['valence'][index_in_hdf5])
            arousal = float(hf['arousal'][index_in_hdf5])
            audio_name = hf['audio_name'][index_in_hdf5].decode()
            
        data_dict = {
            'feature': feature,
            'valence': valence,
            'arousal': arousal,
            'audio_name': audio_name}
            
        return data_dict


class Base(object):
    def __init__(self, indexes_hdf5_path, batch_size):
        """Base class of train sampler and evaluate sampler.
        Args:
            indexes_hdf5_path: string
            batch_size: int
        """
        self.batch_size = batch_size

        with h5py.File(indexes_hdf5_path, 'r') as hf:
            self.audio_indexes = hf['audio_indexes'][:]
            
        self.total_audios = len(self.audio_indexes)
        

class TrainSampler(Base):
    def __init__(self, hdf5_path, holdout_fold, batch_size):
        """Sampler for training.
        Args:
            hdf5_path: string
            holdout_fold: int, e.g., 1
            batch_size: int
        """
        self.hdf5_path = hdf5_path
        
        # Change _waveform to _features in the path
        if 'waveform' in hdf5_path:
            base_path = hdf5_path[:-11]  # Remove 'waveform.h5'
            indexes_hdf5_path = base_path + 'waveform_train.h5'
        else:
            base_path = hdf5_path[:-3]  # Remove '.h5'
            indexes_hdf5_path = base_path + '_train.h5'
            
        super(TrainSampler, self).__init__(indexes_hdf5_path, batch_size)

    def __iter__(self):
        """Generate batch meta for training.
        Returns:
            batch_meta: e.g.: [
                {'hdf5_path': string, 'index_in_hdf5': int}, 
                ...]
        """
        batch_size = self.batch_size

        audio_indexes = self.audio_indexes.copy()
        np.random.shuffle(audio_indexes)

        pointer = 0

        while pointer < len(audio_indexes):
            batch_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            batch_meta = []
            for audio_index in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': audio_index})
            yield batch_meta

    def __len__(self):
        return len(self.audio_indexes) // self.batch_size


class EvaluateSampler(Base):
    def __init__(self, hdf5_path, holdout_fold, batch_size):
        """Sampler for evaluation.
        Args:
            hdf5_path: string
            holdout_fold: int, e.g., 1
            batch_size: int
        """
        self.hdf5_path = hdf5_path
        
        # Change _waveform to _features in the path
        if 'waveform' in hdf5_path:
            base_path = hdf5_path[:-11]  # Remove 'waveform.h5'
            indexes_hdf5_path = base_path + 'waveform_evaluate.h5'
        else:
            base_path = hdf5_path[:-3]  # Remove '.h5'
            indexes_hdf5_path = base_path + '_evaluate.h5'
            
        super(EvaluateSampler, self).__init__(indexes_hdf5_path, batch_size)

    def __iter__(self):
        """Generate batch meta for evaluation.
        Returns:
            batch_meta: e.g.: [
                {'hdf5_path': string, 'index_in_hdf5': int}, 
                ...]
        """
        batch_size = self.batch_size

        audio_indexes = self.audio_indexes.copy()

        pointer = 0

        while pointer < len(audio_indexes):
            batch_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            batch_meta = []
            for audio_index in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': audio_index})
            yield batch_meta

    def __len__(self):
        return len(self.audio_indexes) // self.batch_size


class EmotionTrainSampler(object):
    def __init__(self, hdf5_path, batch_size, train_ratio=0.7):
        """Sampler for emotion training.
        Args:
            hdf5_path: string
            batch_size: int
            train_ratio: float, ratio of training data
        """
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        # Load all audio indexes
        with h5py.File(hdf5_path, 'r') as hf:
            self.audio_indexes = np.arange(len(hf['feature']))
            
        # Split into train and validation
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(self.audio_indexes)
        
        split_idx = int(len(self.audio_indexes) * train_ratio)
        self.train_indexes = self.audio_indexes[:split_idx]
        
        print(f"Training samples: {len(self.train_indexes)}")
        print(f"Total samples: {len(self.audio_indexes)}")

    def __iter__(self):
        """Generate batch meta for training.
        Returns:
            batch_meta: e.g.: [
                {'hdf5_path': string, 'index_in_hdf5': int}, 
                ...]
        """
        batch_size = self.batch_size

        audio_indexes = self.train_indexes.copy()
        np.random.shuffle(audio_indexes)

        pointer = 0

        while pointer < len(audio_indexes):
            batch_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            batch_meta = []
            for audio_index in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': audio_index})
            yield batch_meta

    def __len__(self):
        return len(self.train_indexes) // self.batch_size


class EmotionValidateSampler(object):
    def __init__(self, hdf5_path, batch_size, train_ratio=0.7):
        """Sampler for emotion validation.
        Args:
            hdf5_path: string
            batch_size: int
            train_ratio: float, ratio of training data
        """
        self.hdf5_path = hdf5_path
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        # Load all audio indexes
        with h5py.File(hdf5_path, 'r') as hf:
            self.audio_indexes = np.arange(len(hf['feature']))
            
        # Split into train and validation
        np.random.seed(42)  # For reproducible splits
        np.random.shuffle(self.audio_indexes)
        
        split_idx = int(len(self.audio_indexes) * train_ratio)
        self.validate_indexes = self.audio_indexes[split_idx:]
        
        print(f"Validation samples: {len(self.validate_indexes)}")
        print(f"Total samples: {len(self.audio_indexes)}")

    def __iter__(self):
        """Generate batch meta for validation.
        Returns:
            batch_meta: e.g.: [
                {'hdf5_path': string, 'index_in_hdf5': int}, 
                ...]
        """
        batch_size = self.batch_size

        audio_indexes = self.validate_indexes.copy()

        pointer = 0

        while pointer < len(audio_indexes):
            batch_indexes = audio_indexes[pointer: pointer + batch_size]
            pointer += batch_size

            batch_meta = []
            for audio_index in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': audio_index})
            yield batch_meta

    def __len__(self):
        return len(self.validate_indexes) // self.batch_size