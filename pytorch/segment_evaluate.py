import numpy as np
import logging
from sklearn import metrics
import torch
import torch.nn.functional as F

from pytorch_utils import move_data_to_device
from utilities import get_filename
import config

def calculate_accuracy(y_true, y_score):
    N = y_true.shape[0]
    accuracy = np.sum(np.argmax(y_true, axis=-1) == np.argmax(y_score, axis=-1)) / N
    return accuracy

def forward_features(model, generator, return_target=False):
    """Forward feature data to a model.
    
    Args: 
      model: object
      generator: object
      return_target: bool
    Returns:
      clipwise_output: (segments_num, classes_num)
      (optional) return_target: (segments_num, classes_num)
      segment_idx: (segments_num,)
      audio_name: (segments_num,)
    """
    output_dict = {}
    device = next(model.parameters()).device

    # Forward data to a model in mini-batches
    for n, batch_data_dict in enumerate(generator):
        batch_feature = move_data_to_device(batch_data_dict['feature'], device)
        
        with torch.no_grad():
            model.eval()
            batch_output = model(batch_feature)

        # Append results
        if 'clipwise_output' not in output_dict:
            output_dict['clipwise_output'] = []
        output_dict['clipwise_output'].append(batch_output['clipwise_output'].data.cpu().numpy())
        
        if 'segment_idx' not in output_dict:
            output_dict['segment_idx'] = []
        output_dict['segment_idx'].append(batch_data_dict['segment_idx'].cpu().numpy())
        
        if 'audio_name' not in output_dict:
            output_dict['audio_name'] = []
        output_dict['audio_name'].append(batch_data_dict['audio_name'])
            
        if return_target:
            if 'target' not in output_dict:
                output_dict['target'] = []
            output_dict['target'].append(batch_data_dict['target'].cpu().numpy())

    # Concatenate all results
    for key in output_dict.keys():
        if key == 'audio_name':
            # Flatten the list of lists for audio names
            output_dict[key] = [name for batch in output_dict[key] for name in batch]
        else:
            output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict

class SegmentEvaluator(object):
    def __init__(self, model):
        """
        Evaluator for segment-based predictions.
        Args:
            model: The neural network model
        """
        self.model = model

    def aggregate_predictions_by_audio(self, segment_predictions, audio_names, targets):
        """
        Aggregate predictions from multiple segments of the same audio file.
        Args:
            segment_predictions: numpy array of shape (num_segments, num_classes)
            audio_names: list of audio file names
            targets: numpy array of shape (num_segments, num_classes)
        Returns:
            aggregated_predictions: numpy array of shape (num_audio_files, num_classes)
            file_targets: numpy array of shape (num_audio_files, num_classes)
            unique_audio_names: list of unique audio file names
        """
        unique_audio_names = list(set(audio_names))
        unique_audio_names.sort()  # Sort for consistent ordering
        
        aggregated_predictions = []
        file_targets = []
        
        for audio_name in unique_audio_names:
            # Get all segments for this audio file
            audio_mask = np.array([name == audio_name for name in audio_names])
            file_segments = segment_predictions[audio_mask]
            file_target = targets[audio_mask][0]  # All segments have the same target
            
            # Average the predictions across segments
            aggregated_pred = np.mean(file_segments, axis=0)
            aggregated_predictions.append(aggregated_pred)
            file_targets.append(file_target)
            
        return np.array(aggregated_predictions), np.array(file_targets), unique_audio_names

    def evaluate(self, data_loader):
        """
        Evaluate the model on segmented audio data.
        Args:
            data_loader: DataLoader containing the segmented audio features
        Returns:
            statistics: Dictionary containing evaluation metrics
        """
        # Forward pass
        output_dict = forward_features(
            model=self.model, 
            generator=data_loader, 
            return_target=True)

        clipwise_output = output_dict['clipwise_output']    # (segments_num, classes_num)
        target = output_dict['target']                      # (segments_num, classes_num)
        audio_names = output_dict['audio_name']             # list of audio names

        # Aggregate predictions by audio file
        aggregated_predictions, file_targets, unique_audio_names = self.aggregate_predictions_by_audio(
            clipwise_output, audio_names, target)

        # Calculate metrics
        accuracy = calculate_accuracy(file_targets, aggregated_predictions)
        
        # Calculate confusion matrix
        cm = metrics.confusion_matrix(
            np.argmax(file_targets, axis=-1), 
            np.argmax(aggregated_predictions, axis=-1), 
            labels=None)

        print(f'Evaluated {len(unique_audio_names)} audio files')
        print(f'Audio files: {unique_audio_names}')

        statistics = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'num_audio_files': len(unique_audio_names)
        }

        return statistics 