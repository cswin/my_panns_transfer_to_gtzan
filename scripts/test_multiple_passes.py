#!/usr/bin/env python3

import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import json
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from src.models.emotion_models import FeatureEmotionRegression_Cnn6_LRM
from src.data.data_generator import EmoSoundscapesDataset
from src.data.data_generator import EmotionValidateSampler, emotion_collate_fn
from configs.model_configs import cnn6_config

def test_multiple_passes(model_path, dataset_path, max_passes=6, batch_size=16, num_samples=100):
    """Test emotion feedback model with different numbers of forward passes."""
    print(f"🔬 Testing multiple passes (1 to {max_passes})")
    print(f"📊 Using {num_samples} samples per test")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = FeatureEmotionRegression_Cnn6_LRM(
        sample_rate=32000,
        window_size=cnn6_config['window_size'],
        hop_size=cnn6_config['hop_size'],
        mel_bins=cnn6_config['mel_bins'],
        fmin=cnn6_config['fmin'],
        fmax=cnn6_config['fmax'],
        freeze_base=True,
        forward_passes=2
    )
    
    # Load checkpoint with weights_only=False for compatibility with older checkpoints
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    # Load dataset
    print("Loading dataset...")
    dataset = EmoSoundscapesDataset()
    # Use train_ratio=0.7 to get proper validation split (30% of data)
    validate_sampler = EmotionValidateSampler(hdf5_path=dataset_path, batch_size=batch_size, train_ratio=0.7)
    validate_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_sampler=validate_sampler, 
        collate_fn=emotion_collate_fn, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Collect all data first
    print("Collecting test data...")
    all_features = []
    all_valence_targets = []
    all_arousal_targets = []
    all_audio_names = []
    
    with torch.no_grad():
        for batch_data_dict in tqdm(validate_loader, desc="Loading data"):
            batch_feature = batch_data_dict['feature']
            batch_valence_target = batch_data_dict['valence']
            batch_arousal_target = batch_data_dict['arousal']
            batch_audio_name = batch_data_dict['audio_name']
            
            # Add channel dimension if needed
            if len(batch_feature.shape) == 3:
                batch_feature = batch_feature.unsqueeze(1)
            
            all_features.append(batch_feature)
            all_valence_targets.append(batch_valence_target)
            all_arousal_targets.append(batch_arousal_target)
            all_audio_names.extend(batch_audio_name)
    
    # Concatenate all data
    all_features = torch.cat(all_features, dim=0)
    all_valence_targets = torch.cat(all_valence_targets, dim=0)
    all_arousal_targets = torch.cat(all_arousal_targets, dim=0)
    
    print(f"Total samples loaded: {len(all_features)}")
    
    # Sample subset for testing
    if num_samples == 0:
        # Use all samples (full validation set)
        test_features = all_features
        test_valence_targets = all_valence_targets
        test_arousal_targets = all_arousal_targets
        test_audio_names = all_audio_names
        print(f"Using ALL validation samples: {len(test_features)}")
    elif num_samples < len(all_features):
        # Sample a subset
        indices = np.random.choice(len(all_features), num_samples, replace=False)
        test_features = all_features[indices]
        test_valence_targets = all_valence_targets[indices]
        test_arousal_targets = all_arousal_targets[indices]
        test_audio_names = [all_audio_names[i] for i in indices]
        print(f"Using subset of {len(test_features)} samples")
    else:
        # num_samples >= total samples, use all
        test_features = all_features
        test_valence_targets = all_valence_targets
        test_arousal_targets = all_arousal_targets
        test_audio_names = all_audio_names
        print(f"Using all {len(test_features)} samples (num_samples >= total)")
    
    print(f"Testing with {len(test_features)} samples")
    
    # Test different numbers of passes
    results = []
    
    for num_passes in range(1, max_passes + 1):
        print(f"\n🔄 Testing {num_passes} forward pass(es)...")
        
        valence_predictions = []
        arousal_predictions = []
        processing_times = []
        
        with torch.no_grad():
            for i in tqdm(range(len(test_features)), desc=f"Pass {num_passes}", leave=False):
                # Clear feedback state for each sample
                if hasattr(model, 'lrm'):
                    model.lrm.clear_stored_activations()
                
                # Prepare input
                sample = test_features[i:i+1].to(device)
                
                # Time the forward pass
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                output = model(sample, forward_passes=num_passes)
                end_time.record()
                
                torch.cuda.synchronize()
                processing_time = start_time.elapsed_time(end_time)
                
                # Get predictions
                valence_pred = output['valence'].cpu().numpy()[0, 0]
                arousal_pred = output['arousal'].cpu().numpy()[0, 0]
                
                valence_predictions.append(valence_pred)
                arousal_predictions.append(arousal_pred)
                processing_times.append(processing_time)
        
        # Calculate metrics
        valence_predictions = np.array(valence_predictions)
        arousal_predictions = np.array(arousal_predictions)
        valence_targets = test_valence_targets.numpy()
        arousal_targets = test_arousal_targets.numpy()
        
        # Valence metrics
        valence_mae = mean_absolute_error(valence_targets, valence_predictions)
        valence_rmse = np.sqrt(mean_squared_error(valence_targets, valence_predictions))
        valence_pearson = pearsonr(valence_targets, valence_predictions)[0]
        valence_spearman = spearmanr(valence_targets, valence_predictions)[0]
        
        # Arousal metrics
        arousal_mae = mean_absolute_error(arousal_targets, arousal_predictions)
        arousal_rmse = np.sqrt(mean_squared_error(arousal_targets, arousal_predictions))
        arousal_pearson = pearsonr(arousal_targets, arousal_predictions)[0]
        arousal_spearman = spearmanr(arousal_targets, arousal_predictions)[0]
        
        # Combined metrics
        mean_mae = (valence_mae + arousal_mae) / 2
        mean_rmse = (valence_rmse + arousal_rmse) / 2
        mean_pearson = (valence_pearson + arousal_pearson) / 2
        mean_spearman = (valence_spearman + arousal_spearman) / 2
        
        # Timing
        avg_processing_time = np.mean(processing_times)
        
        # Store results
        result = {
            'num_passes': num_passes,
            'valence_mae': valence_mae,
            'valence_rmse': valence_rmse,
            'valence_pearson': valence_pearson,
            'valence_spearman': valence_spearman,
            'arousal_mae': arousal_mae,
            'arousal_rmse': arousal_rmse,
            'arousal_pearson': arousal_pearson,
            'arousal_spearman': arousal_spearman,
            'mean_mae': mean_mae,
            'mean_rmse': mean_rmse,
            'mean_pearson': mean_pearson,
            'mean_spearman': mean_spearman,
            'avg_processing_time_ms': avg_processing_time,
            'total_processing_time_ms': np.sum(processing_times)
        }
        
        results.append(result)
        
        print(f"  ✅ {num_passes} pass(es) - Mean MAE: {mean_mae:.4f}, Mean Pearson: {mean_pearson:.4f}")
        print(f"      Valence: MAE={valence_mae:.4f}, Pearson={valence_pearson:.4f}")
        print(f"      Arousal: MAE={arousal_mae:.4f}, Pearson={arousal_pearson:.4f}")
        print(f"      Avg time: {avg_processing_time:.2f}ms")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df

def plot_results(results_df, output_dir):
    """Plot the results of multiple pass testing."""
    print("\n📊 Creating plots...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Emotion Feedback Model: Performance vs Number of Forward Passes', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean MAE
    axes[0, 0].plot(results_df['num_passes'], results_df['mean_mae'], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_title('Mean MAE vs Passes')
    axes[0, 0].set_xlabel('Number of Forward Passes')
    axes[0, 0].set_ylabel('Mean Absolute Error')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Mean RMSE
    axes[0, 1].plot(results_df['num_passes'], results_df['mean_rmse'], 'o-', linewidth=2, markersize=8, color='orange')
    axes[0, 1].set_title('Mean RMSE vs Passes')
    axes[0, 1].set_xlabel('Number of Forward Passes')
    axes[0, 1].set_ylabel('Root Mean Square Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Mean Pearson Correlation
    axes[0, 2].plot(results_df['num_passes'], results_df['mean_pearson'], 'o-', linewidth=2, markersize=8, color='green')
    axes[0, 2].set_title('Mean Pearson Correlation vs Passes')
    axes[0, 2].set_xlabel('Number of Forward Passes')
    axes[0, 2].set_ylabel('Pearson Correlation')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Separate Valence/Arousal MAE
    axes[1, 0].plot(results_df['num_passes'], results_df['valence_mae'], 'o-', linewidth=2, markersize=8, label='Valence', color='blue')
    axes[1, 0].plot(results_df['num_passes'], results_df['arousal_mae'], 's-', linewidth=2, markersize=8, label='Arousal', color='red')
    axes[1, 0].set_title('Valence vs Arousal MAE')
    axes[1, 0].set_xlabel('Number of Forward Passes')
    axes[1, 0].set_ylabel('Mean Absolute Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Separate Valence/Arousal Pearson
    axes[1, 1].plot(results_df['num_passes'], results_df['valence_pearson'], 'o-', linewidth=2, markersize=8, label='Valence', color='blue')
    axes[1, 1].plot(results_df['num_passes'], results_df['arousal_pearson'], 's-', linewidth=2, markersize=8, label='Arousal', color='red')
    axes[1, 1].set_title('Valence vs Arousal Pearson Correlation')
    axes[1, 1].set_xlabel('Number of Forward Passes')
    axes[1, 1].set_ylabel('Pearson Correlation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Processing Time
    axes[1, 2].plot(results_df['num_passes'], results_df['avg_processing_time_ms'], 'o-', linewidth=2, markersize=8, color='purple')
    axes[1, 2].set_title('Average Processing Time vs Passes')
    axes[1, 2].set_xlabel('Number of Forward Passes')
    axes[1, 2].set_ylabel('Processing Time (ms)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'multiple_passes_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📈 Plot saved to: {plot_path}")
    
    # Create summary plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot key metrics
    ax.plot(results_df['num_passes'], results_df['mean_mae'], 'o-', linewidth=3, markersize=10, label='Mean MAE', color='red')
    ax.plot(results_df['num_passes'], results_df['mean_pearson'], 's-', linewidth=3, markersize=10, label='Mean Pearson', color='blue')
    
    # Add processing time on secondary y-axis
    ax2 = ax.twinx()
    ax2.plot(results_df['num_passes'], results_df['avg_processing_time_ms'], '^-', linewidth=2, markersize=8, label='Processing Time (ms)', color='green', alpha=0.7)
    
    ax.set_xlabel('Number of Forward Passes', fontsize=12)
    ax.set_ylabel('Performance Metrics', fontsize=12)
    ax2.set_ylabel('Processing Time (ms)', fontsize=12)
    ax.set_title('Emotion Feedback Model: Performance vs Computational Cost', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.tight_layout()
    
    # Save summary plot
    summary_plot_path = os.path.join(output_dir, 'multiple_passes_summary.png')
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 Summary plot saved to: {summary_plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Test multiple passes for emotion feedback model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to emotion features HDF5 file')
    parser.add_argument('--max-passes', type=int, default=6, help='Maximum number of passes to test')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of samples to test per pass')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Test multiple passes
    results_df = test_multiple_passes(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        max_passes=args.max_passes,
        batch_size=args.batch_size,
        num_samples=args.num_samples
    )
    
    # Save results
    results_path = os.path.join(args.output_dir, 'multiple_passes_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Save results as JSON for easy access
    json_path = os.path.join(args.output_dir, 'multiple_passes_results.json')
    results_df.to_json(json_path, orient='records', indent=2)
    print(f"📄 JSON results saved to: {json_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 MULTIPLE PASSES TESTING SUMMARY")
    print("=" * 60)
    
    best_mae_idx = results_df['mean_mae'].idxmin()
    best_pearson_idx = results_df['mean_pearson'].idxmax()
    
    print(f"🎯 Best MAE: {results_df.loc[best_mae_idx, 'mean_mae']:.4f} at {results_df.loc[best_mae_idx, 'num_passes']} passes")
    print(f"�� Best Pearson: {results_df.loc[best_pearson_idx, 'mean_pearson']:.4f} at {results_df.loc[best_pearson_idx, 'num_passes']} passes")
    
    print("\n📈 Performance vs Passes:")
    for _, row in results_df.iterrows():
        print(f"  {row['num_passes']} pass(es): MAE={row['mean_mae']:.4f}, Pearson={row['mean_pearson']:.4f}, Time={row['avg_processing_time_ms']:.1f}ms")
    
    # Create plots
    plot_results(results_df, args.output_dir)
    
    print(f"\n✅ Multiple passes testing completed!")
    print(f"📁 All results saved in: {args.output_dir}")

if __name__ == '__main__':
    main()
