import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time


class EmotionEvaluator(object):
    def __init__(self, model):
        """Evaluator for emotion regression models.
        
        Args:
            model: torch model for emotion prediction
        """
        self.model = model
        
    def evaluate(self, data_loader):
        """Evaluate the model and return emotion prediction metrics.
        
        Args:
            data_loader: torch data loader
            
        Returns:
            statistics: dict of evaluation metrics
        """
        # Generate predictions
        output_dict = self._forward(data_loader)
        
        # Calculate regression metrics
        statistics = self._calculate_metrics(output_dict)
        
        return statistics
    
    def _forward(self, data_loader):
        """Forward data to model and get predictions.
        
        Args:
            data_loader: torch data loader
            
        Returns:
            output_dict: dict containing predictions and targets
        """
        self.model.eval()
        
        output_dict = {
            'valence_pred': [],
            'arousal_pred': [],
            'valence_target': [],
            'arousal_target': [],
            'audio_name': []
        }
        
        with torch.no_grad():
            for batch_data_dict in data_loader:
                
                batch_feature = batch_data_dict['feature']
                batch_valence_target = batch_data_dict['valence']
                batch_arousal_target = batch_data_dict['arousal']
                batch_audio_name = batch_data_dict['audio_name']
                
                # Add channel dimension if needed for feature input
                if len(batch_feature.shape) == 3:  # (batch_size, time_steps, mel_bins)
                    batch_feature = batch_feature.unsqueeze(1)  # (batch_size, 1, time_steps, mel_bins)
                
                batch_feature = batch_feature.cuda()
                
                # Forward pass
                batch_output_dict = self.model(batch_feature, None)
                
                # Move predictions to CPU
                batch_valence_pred = batch_output_dict['valence'].detach().cpu().numpy()
                batch_arousal_pred = batch_output_dict['arousal'].detach().cpu().numpy()
                
                # Collect predictions and targets
                output_dict['valence_pred'].extend(batch_valence_pred.flatten())
                output_dict['arousal_pred'].extend(batch_arousal_pred.flatten())
                output_dict['valence_target'].extend(batch_valence_target.numpy())
                output_dict['arousal_target'].extend(batch_arousal_target.numpy())
                output_dict['audio_name'].extend(batch_audio_name)
                
        # Convert to numpy arrays
        for key in ['valence_pred', 'arousal_pred', 'valence_target', 'arousal_target']:
            output_dict[key] = np.array(output_dict[key])
            
        return output_dict
    
    def _calculate_metrics(self, output_dict):
        """Calculate regression metrics for emotion prediction.
        
        Args:
            output_dict: dict containing predictions and targets
            
        Returns:
            statistics: dict of metrics
        """
        valence_pred = output_dict['valence_pred']
        arousal_pred = output_dict['arousal_pred']
        valence_target = output_dict['valence_target']
        arousal_target = output_dict['arousal_target']
        audio_names = output_dict['audio_name']
        
        # Calculate both segment-level and audio-level metrics
        statistics = {}
        
        # Segment-level metrics (current approach)
        statistics.update(self._calculate_segment_metrics(
            valence_pred, arousal_pred, valence_target, arousal_target, prefix='segment_'))
        
        # Audio-level metrics (aggregated by audio file)
        statistics.update(self._calculate_audio_metrics(
            valence_pred, arousal_pred, valence_target, arousal_target, audio_names, prefix='audio_'))
        
        return statistics
    
    def _calculate_segment_metrics(self, valence_pred, arousal_pred, valence_target, arousal_target, prefix=''):
        """Calculate metrics at segment level."""
        statistics = {}
        
        # Mean Squared Error
        statistics[f'{prefix}valence_mse'] = mean_squared_error(valence_target, valence_pred)
        statistics[f'{prefix}arousal_mse'] = mean_squared_error(arousal_target, arousal_pred)
        statistics[f'{prefix}mean_mse'] = (statistics[f'{prefix}valence_mse'] + statistics[f'{prefix}arousal_mse']) / 2
        
        # Root Mean Squared Error
        statistics[f'{prefix}valence_rmse'] = np.sqrt(statistics[f'{prefix}valence_mse'])
        statistics[f'{prefix}arousal_rmse'] = np.sqrt(statistics[f'{prefix}arousal_mse'])
        statistics[f'{prefix}mean_rmse'] = (statistics[f'{prefix}valence_rmse'] + statistics[f'{prefix}arousal_rmse']) / 2
        
        # Mean Absolute Error
        statistics[f'{prefix}valence_mae'] = mean_absolute_error(valence_target, valence_pred)
        statistics[f'{prefix}arousal_mae'] = mean_absolute_error(arousal_target, arousal_pred)
        statistics[f'{prefix}mean_mae'] = (statistics[f'{prefix}valence_mae'] + statistics[f'{prefix}arousal_mae']) / 2
        
        # Pearson Correlation
        valence_pearson, valence_p_pearson = pearsonr(valence_target, valence_pred)
        arousal_pearson, arousal_p_pearson = pearsonr(arousal_target, arousal_pred)
        statistics[f'{prefix}valence_pearson'] = valence_pearson
        statistics[f'{prefix}arousal_pearson'] = arousal_pearson
        statistics[f'{prefix}mean_pearson'] = (valence_pearson + arousal_pearson) / 2
        
        # Spearman Correlation
        valence_spearman, valence_p_spearman = spearmanr(valence_target, valence_pred)
        arousal_spearman, arousal_p_spearman = spearmanr(arousal_target, arousal_pred)
        statistics[f'{prefix}valence_spearman'] = valence_spearman
        statistics[f'{prefix}arousal_spearman'] = arousal_spearman
        statistics[f'{prefix}mean_spearman'] = (valence_spearman + arousal_spearman) / 2
        
        # R-squared (coefficient of determination)
        valence_r2 = 1 - (np.sum((valence_target - valence_pred) ** 2) / 
                         np.sum((valence_target - np.mean(valence_target)) ** 2))
        arousal_r2 = 1 - (np.sum((arousal_target - arousal_pred) ** 2) / 
                         np.sum((arousal_target - np.mean(arousal_target)) ** 2))
        statistics[f'{prefix}valence_r2'] = valence_r2
        statistics[f'{prefix}arousal_r2'] = arousal_r2
        statistics[f'{prefix}mean_r2'] = (valence_r2 + arousal_r2) / 2
        
        # Additional statistics
        statistics[f'{prefix}num_samples'] = len(valence_target)
        statistics[f'{prefix}valence_target_mean'] = np.mean(valence_target)
        statistics[f'{prefix}valence_target_std'] = np.std(valence_target)
        statistics[f'{prefix}arousal_target_mean'] = np.mean(arousal_target)
        statistics[f'{prefix}arousal_target_std'] = np.std(arousal_target)
        statistics[f'{prefix}valence_pred_mean'] = np.mean(valence_pred)
        statistics[f'{prefix}valence_pred_std'] = np.std(valence_pred)
        statistics[f'{prefix}arousal_pred_mean'] = np.mean(arousal_pred)
        statistics[f'{prefix}arousal_pred_std'] = np.std(arousal_pred)
        
        return statistics
    
    def _calculate_audio_metrics(self, valence_pred, arousal_pred, valence_target, arousal_target, audio_names, prefix=''):
        """Calculate metrics at audio file level by aggregating segments."""
        
        # Group predictions and targets by base audio file
        def get_base_name(name):
            name_str = name.decode() if isinstance(name, bytes) else name
            return name_str.split('_seg')[0] if '_seg' in name_str else name_str
        
        audio_data = {}
        for i, name in enumerate(audio_names):
            base_name = get_base_name(name)
            if base_name not in audio_data:
                audio_data[base_name] = {
                    'valence_pred': [],
                    'arousal_pred': [],
                    'valence_target': [],
                    'arousal_target': []
                }
            audio_data[base_name]['valence_pred'].append(valence_pred[i])
            audio_data[base_name]['arousal_pred'].append(arousal_pred[i])
            audio_data[base_name]['valence_target'].append(valence_target[i])
            audio_data[base_name]['arousal_target'].append(arousal_target[i])
        
        # Aggregate predictions for each audio file (mean of segments)
        audio_valence_pred = []
        audio_arousal_pred = []
        audio_valence_target = []
        audio_arousal_target = []
        
        for base_name, data in audio_data.items():
            # Average predictions across segments
            audio_valence_pred.append(np.mean(data['valence_pred']))
            audio_arousal_pred.append(np.mean(data['arousal_pred']))
            # Target should be the same for all segments, but take mean to be safe
            audio_valence_target.append(np.mean(data['valence_target']))
            audio_arousal_target.append(np.mean(data['arousal_target']))
        
        # Convert to numpy arrays
        audio_valence_pred = np.array(audio_valence_pred)
        audio_arousal_pred = np.array(audio_arousal_pred)
        audio_valence_target = np.array(audio_valence_target)
        audio_arousal_target = np.array(audio_arousal_target)
        
        # Calculate metrics on aggregated audio-level predictions
        return self._calculate_segment_metrics(
            audio_valence_pred, audio_arousal_pred, 
            audio_valence_target, audio_arousal_target, prefix)
    
    def print_evaluation(self, statistics):
        """Print evaluation results in a formatted way."""
        
        # Print segment-level metrics
        print(f"Number of segments: {statistics['segment_num_samples']}")
        print("\n=== SEGMENT-LEVEL METRICS ===")
        print("=== Valence Metrics ===")
        print(f"MSE:      {statistics['segment_valence_mse']:.4f}")
        print(f"RMSE:     {statistics['segment_valence_rmse']:.4f}")
        print(f"MAE:      {statistics['segment_valence_mae']:.4f}")
        print(f"Pearson:  {statistics['segment_valence_pearson']:.4f}")
        print(f"Spearman: {statistics['segment_valence_spearman']:.4f}")
        print(f"R²:       {statistics['segment_valence_r2']:.4f}")
        
        print("\n=== Arousal Metrics ===")
        print(f"MSE:      {statistics['segment_arousal_mse']:.4f}")
        print(f"RMSE:     {statistics['segment_arousal_rmse']:.4f}")
        print(f"MAE:      {statistics['segment_arousal_mae']:.4f}")
        print(f"Pearson:  {statistics['segment_arousal_pearson']:.4f}")
        print(f"Spearman: {statistics['segment_arousal_spearman']:.4f}")
        print(f"R²:       {statistics['segment_arousal_r2']:.4f}")
        
        print("\n=== Average Metrics ===")
        print(f"Mean MSE:      {statistics['segment_mean_mse']:.4f}")
        print(f"Mean RMSE:     {statistics['segment_mean_rmse']:.4f}")
        print(f"Mean MAE:      {statistics['segment_mean_mae']:.4f}")
        print(f"Mean Pearson:  {statistics['segment_mean_pearson']:.4f}")
        print(f"Mean Spearman: {statistics['segment_mean_spearman']:.4f}")
        print(f"Mean R²:       {statistics['segment_mean_r2']:.4f}")
        
        # Print audio-level metrics
        print(f"\n\nNumber of audio files: {statistics['audio_num_samples']}")
        print("\n=== AUDIO-LEVEL METRICS (Aggregated) ===")
        print("=== Valence Metrics ===")
        print(f"MSE:      {statistics['audio_valence_mse']:.4f}")
        print(f"RMSE:     {statistics['audio_valence_rmse']:.4f}")
        print(f"MAE:      {statistics['audio_valence_mae']:.4f}")
        print(f"Pearson:  {statistics['audio_valence_pearson']:.4f}")
        print(f"Spearman: {statistics['audio_valence_spearman']:.4f}")
        print(f"R²:       {statistics['audio_valence_r2']:.4f}")
        
        print("\n=== Arousal Metrics ===")
        print(f"MSE:      {statistics['audio_arousal_mse']:.4f}")
        print(f"RMSE:     {statistics['audio_arousal_rmse']:.4f}")
        print(f"MAE:      {statistics['audio_arousal_mae']:.4f}")
        print(f"Pearson:  {statistics['audio_arousal_pearson']:.4f}")
        print(f"Spearman: {statistics['audio_arousal_spearman']:.4f}")
        print(f"R²:       {statistics['audio_arousal_r2']:.4f}")
        
        print("\n=== Average Metrics ===")
        print(f"Mean MSE:      {statistics['audio_mean_mse']:.4f}")
        print(f"Mean RMSE:     {statistics['audio_mean_rmse']:.4f}")
        print(f"Mean MAE:      {statistics['audio_mean_mae']:.4f}")
        print(f"Mean Pearson:  {statistics['audio_mean_pearson']:.4f}")
        print(f"Mean Spearman: {statistics['audio_mean_spearman']:.4f}")
        print(f"Mean R²:       {statistics['audio_mean_r2']:.4f}") 