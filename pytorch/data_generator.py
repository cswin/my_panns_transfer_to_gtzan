import numpy as np
import h5py
import torch


def collate_fn(list_data_dict):
    """
    Collate input features and targets of segments.
    Args:
        list_data_dict: list of dict, each dict contains:
            'feature': (time_steps, mel_bins)
            'target': (classes_num,)
            'segment_idx': int
            'audio_name': str
    Returns:
        np_data_dict: dict containing:
            'feature': (batch_size, time_steps, mel_bins)
            'target': (batch_size, classes_num)
            'segment_idx': (batch_size,)
            'audio_name': list of str
    """
    np_data_dict = {}
    
    # Stack features
    features = np.array([data_dict['feature'] for data_dict in list_data_dict])
    np_data_dict['feature'] = torch.Tensor(features)
    
    # Stack targets
    targets = np.array([data_dict['target'] for data_dict in list_data_dict])
    np_data_dict['target'] = torch.Tensor(targets)
    
    # Stack segment indices
    segment_indices = np.array([data_dict['segment_idx'] for data_dict in list_data_dict])
    np_data_dict['segment_idx'] = torch.LongTensor(segment_indices)
    
    # Collect audio names
    audio_names = [data_dict['audio_name'] for data_dict in list_data_dict]
    np_data_dict['audio_name'] = audio_names
    
    return np_data_dict


class GtzanDataset(object):
    def __init__(self):
        """This class takes the meta of an audio segment as input, and return 
        the feature and target of the audio segment. This class is used by DataLoader.
        """
        pass
        
    def __getitem__(self, meta):
        """Load feature and target of an audio segment.
        Args:
            meta: {
                'hdf5_path': str, 
                'index_in_hdf5': int}
        Returns:
            data_dict: {
                'feature': (time_steps, mel_bins)
                'target': (classes_num,),
                'segment_idx': int,
                'audio_name': str
            }
        """
        hdf5_path = meta['hdf5_path']
        index_in_hdf5 = meta['index_in_hdf5']
        
        with h5py.File(hdf5_path, 'r') as hf:
            feature = hf['feature'][index_in_hdf5]
            target = hf['target'][index_in_hdf5]
            segment_idx = hf['segment_idx'][index_in_hdf5]
            audio_name = hf['audio_name'][index_in_hdf5].decode()
            
        data_dict = {
            'feature': feature,
            'target': target,
            'segment_idx': segment_idx,
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
            self.segment_indexes = hf['segment_indexes'][:]
            
        self.total_segments = len(self.segment_indexes)
        

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

        n = len(self.segment_indexes)
        indexes = np.array(self.segment_indexes)

        np.random.shuffle(indexes)
        
        pointer = 0
        while pointer < n:
            batch_indexes = indexes[pointer : pointer + batch_size]
            pointer += batch_size

            batch_meta = []
            for index in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': index})
                    
            yield batch_meta

    def __len__(self):
        return -1


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
            indexes_hdf5_path = base_path + 'waveform_validate.h5'
        else:
            base_path = hdf5_path[:-3]  # Remove '.h5'
            indexes_hdf5_path = base_path + '_validate.h5'
            
        super(EvaluateSampler, self).__init__(indexes_hdf5_path, batch_size)

    def __iter__(self):
        """Generate batch meta for evaluation.
        Returns:
            batch_meta: e.g.: [
                {'hdf5_path': string, 'index_in_hdf5': int}, 
                ...]
        """
        batch_size = self.batch_size
        
        n = len(self.segment_indexes)
        indexes = np.array(self.segment_indexes)
        
        pointer = 0
        while pointer < n:
            batch_indexes = indexes[pointer : pointer + batch_size]
            pointer += batch_size
            
            batch_meta = []
            for index in batch_indexes:
                batch_meta.append({
                    'hdf5_path': self.hdf5_path, 
                    'index_in_hdf5': index})
                    
            yield batch_meta
            
    def __len__(self):
        return -1 