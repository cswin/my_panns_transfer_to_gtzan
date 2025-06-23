import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import pandas as pd
import os
from collections import defaultdict
from src.training.evaluator import EmotionEvaluator


class LRMEmotionEvaluator(EmotionEvaluator):
    """
    LRM-aware evaluator for emotion regression models.
    
    This evaluator handles segment-based processing where segments from the same
    audio file are processed sequentially to enable feedback between segments.
    """
    
    def __init__(self, model):
        """Evaluator for LRM emotion regression models.
        
        Args:
            model: torch LRM model for emotion prediction
        """
        super().__init__(model)
        
    def _forward(self, data_loader):
        """Forward data to LRM model with segment-based feedback.
        
        For LRM models, we need to process segments from the same audio file
        sequentially so that feedback from segment N can modulate segment N+1.
        
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
        
        # Group data by base audio file for sequential processing
        audio_segments = defaultdict(list)
        
        # First pass: collect all data and group by audio file
        print("Collecting and grouping segments by audio file...")
        with torch.no_grad():
            for batch_data_dict in data_loader:
                batch_feature = batch_data_dict['feature']
                batch_valence_target = batch_data_dict['valence']
                batch_arousal_target = batch_data_dict['arousal']
                batch_audio_name = batch_data_dict['audio_name']
                
                # Add channel dimension if needed for feature input
                if len(batch_feature.shape) == 3:  # (batch_size, time_steps, mel_bins)
                    batch_feature = batch_feature.unsqueeze(1)  # (batch_size, 1, time_steps, mel_bins)
                
                # Group by base audio file
                for i in range(len(batch_audio_name)):
                    audio_name = batch_audio_name[i]
                    base_name = self._get_base_audio_name(audio_name)
                    segment_idx = self._get_segment_index(audio_name)
                    
                    audio_segments[base_name].append({
                        'feature': batch_feature[i:i+1],  # Keep batch dimension
                        'valence_target': batch_valence_target[i],
                        'arousal_target': batch_arousal_target[i],
                        'audio_name': audio_name,
                        'segment_idx': segment_idx
                    })
        
        # Sort segments within each audio file by segment index
        for base_name in audio_segments:
            audio_segments[base_name].sort(key=lambda x: x['segment_idx'])
        
        print(f"Processing {len(audio_segments)} audio files with segment-based feedback...")
        
        # Second pass: process each audio file's segments sequentially
        with torch.no_grad():
            for audio_idx, (base_name, segments) in enumerate(audio_segments.items()):
                if audio_idx % 50 == 0:
                    print(f"Processing audio {audio_idx+1}/{len(audio_segments)}: {base_name}")
                
                # Clear any previous feedback state for new audio file
                if hasattr(self.model, 'lrm'):
                    self.model.lrm.clear_stored_activations()
                
                # Process segments sequentially for this audio file
                for seg_idx, segment_data in enumerate(segments):
                    feature = segment_data['feature'].cuda()
                    
                    # For first segment: no feedback (pure feedforward)
                    # For subsequent segments: use feedback from previous segment
                    if seg_idx == 0:
                        # First segment: pure feedforward pass
                        output = self.model(feature, None, 2)  # (input, mixup_lambda, forward_passes) - NEED 2 passes for feedback!
                    else:
                        # Subsequent segments: use feedback from previous segment
                        # The model should have stored activations from the previous segment
                        output = self.model(feature, None, 2)  # (input, mixup_lambda, forward_passes) - NEED 2 passes for feedback!
                    
                    # Store predictions for this segment
                    valence_pred = output['valence'].detach().cpu().numpy().flatten()[0]
                    arousal_pred = output['arousal'].detach().cpu().numpy().flatten()[0]
                    
                    output_dict['valence_pred'].append(valence_pred)
                    output_dict['arousal_pred'].append(arousal_pred)
                    output_dict['valence_target'].append(segment_data['valence_target'].numpy())
                    output_dict['arousal_target'].append(segment_data['arousal_target'].numpy())
                    output_dict['audio_name'].append(segment_data['audio_name'])
                    
                    # Store feedback for next segment (if not the last segment)
                    if seg_idx < len(segments) - 1:
                        # Extract embedding-based modulation signals for next segment
                        embedding = output['embedding']
                        
                        if hasattr(self.model, 'embedding_valence_transform'):
                            valence_modulation = self.model.embedding_valence_transform(embedding)
                            arousal_modulation = self.model.embedding_arousal_transform(embedding)
                            
                            # Store modulation signals for next segment
                            for conn in self.model.lrm.mod_connections:
                                source_name = conn['source']
                                target_name = conn['target']
                                
                                if source_name == 'embedding_valence':
                                    self.model.lrm.store_source_activation(
                                        valence_modulation, source_name, target_name
                                    )
                                elif source_name == 'embedding_arousal':
                                    self.model.lrm.store_source_activation(
                                        arousal_modulation, source_name, target_name
                                    )
        
        # Convert to numpy arrays
        for key in ['valence_pred', 'arousal_pred', 'valence_target', 'arousal_target']:
            output_dict[key] = np.array(output_dict[key])
            
        print(f"Processed {len(output_dict['valence_pred'])} segments total")
        return output_dict
    
    def _get_base_audio_name(self, audio_name):
        """Extract base audio name from segment name."""
        name_str = audio_name.decode() if isinstance(audio_name, bytes) else audio_name
        return name_str.split('_seg')[0] if '_seg' in name_str else name_str
    
    def _get_segment_index(self, audio_name):
        """Extract segment index from segment name."""
        name_str = audio_name.decode() if isinstance(audio_name, bytes) else audio_name
        if '_seg' in name_str:
            try:
                return int(name_str.split('_seg')[1])
            except (IndexError, ValueError):
                return 0
        return 0
    
    def evaluate_with_feedback_analysis(self, data_loader, save_predictions=False, output_dir=None):
        """
        Evaluate the LRM model with detailed feedback analysis.
        
        This method provides additional insights into how feedback affects predictions
        across segments within the same audio file.
        
        Args:
            data_loader: torch data loader
            save_predictions: bool, whether to save predictions to CSV files
            output_dir: str, directory to save CSV files
            
        Returns:
            statistics: dict of evaluation metrics
            feedback_analysis: dict of feedback-specific metrics
        """
        # Generate predictions with segment-based feedback
        output_dict = self._forward(data_loader)
        
        # Save predictions if requested
        if save_predictions:
            if output_dir is None:
                raise ValueError("output_dir must be provided when save_predictions=True")
            os.makedirs(output_dir, exist_ok=True)
            self._save_predictions_to_csv(output_dict, output_dir)
        
        # Calculate standard regression metrics
        statistics = self._calculate_metrics(output_dict)
        
        # Calculate feedback-specific analysis
        feedback_analysis = self._analyze_feedback_effects(output_dict)
        
        return statistics, feedback_analysis
    
    def _analyze_feedback_effects(self, output_dict):
        """
        Analyze how feedback affects predictions across segments.
        
        Args:
            output_dict: dict containing predictions and targets
            
        Returns:
            feedback_analysis: dict of feedback-specific metrics
        """
        # Group predictions by audio file and segment
        audio_data = defaultdict(list)
        
        for i, audio_name in enumerate(output_dict['audio_name']):
            base_name = self._get_base_audio_name(audio_name)
            segment_idx = self._get_segment_index(audio_name)
            
            audio_data[base_name].append({
                'segment_idx': segment_idx,
                'valence_pred': output_dict['valence_pred'][i],
                'arousal_pred': output_dict['arousal_pred'][i],
                'valence_target': output_dict['valence_target'][i],
                'arousal_target': output_dict['arousal_target'][i]
            })
        
        # Sort segments within each audio file
        for base_name in audio_data:
            audio_data[base_name].sort(key=lambda x: x['segment_idx'])
        
        # Analyze feedback effects
        feedback_analysis = {
            'num_audio_files': len(audio_data),
            'avg_segments_per_audio': np.mean([len(segments) for segments in audio_data.values()]),
            'segment_position_effects': {},
            'prediction_consistency': {}
        }
        
        # Analyze performance by segment position
        segment_positions = defaultdict(list)
        for segments in audio_data.values():
            for seg_data in segments:
                pos = seg_data['segment_idx']
                segment_positions[pos].append({
                    'valence_error': abs(seg_data['valence_pred'] - seg_data['valence_target']),
                    'arousal_error': abs(seg_data['arousal_pred'] - seg_data['arousal_target'])
                })
        
        for pos, errors in segment_positions.items():
            if len(errors) > 0:
                feedback_analysis['segment_position_effects'][f'segment_{pos}'] = {
                    'count': len(errors),
                    'valence_mae': np.mean([e['valence_error'] for e in errors]),
                    'arousal_mae': np.mean([e['arousal_error'] for e in errors])
                }
        
        # Analyze prediction consistency within audio files
        valence_consistencies = []
        arousal_consistencies = []
        
        for segments in audio_data.values():
            if len(segments) > 1:
                valence_preds = [s['valence_pred'] for s in segments]
                arousal_preds = [s['arousal_pred'] for s in segments]
                
                # Calculate standard deviation as measure of consistency
                valence_consistencies.append(np.std(valence_preds))
                arousal_consistencies.append(np.std(arousal_preds))
        
        if valence_consistencies:
            feedback_analysis['prediction_consistency'] = {
                'valence_std_mean': np.mean(valence_consistencies),
                'valence_std_std': np.std(valence_consistencies),
                'arousal_std_mean': np.mean(arousal_consistencies),
                'arousal_std_std': np.std(arousal_consistencies)
            }
        
        return feedback_analysis
    
    def print_feedback_analysis(self, feedback_analysis):
        """Print feedback analysis results."""
        print("\n=== LRM FEEDBACK ANALYSIS ===")
        print(f"Number of audio files: {feedback_analysis['num_audio_files']}")
        print(f"Average segments per audio: {feedback_analysis['avg_segments_per_audio']:.1f}")
        
        print("\n--- Performance by Segment Position ---")
        for pos_key, metrics in feedback_analysis['segment_position_effects'].items():
            print(f"{pos_key}: Count={metrics['count']}, "
                  f"Valence MAE={metrics['valence_mae']:.4f}, "
                  f"Arousal MAE={metrics['arousal_mae']:.4f}")
        
        if 'prediction_consistency' in feedback_analysis:
            print("\n--- Prediction Consistency Within Audio Files ---")
            consistency = feedback_analysis['prediction_consistency']
            print(f"Valence std (mean±std): {consistency['valence_std_mean']:.4f}±{consistency['valence_std_std']:.4f}")
            print(f"Arousal std (mean±std): {consistency['arousal_std_mean']:.4f}±{consistency['arousal_std_std']:.4f}") 