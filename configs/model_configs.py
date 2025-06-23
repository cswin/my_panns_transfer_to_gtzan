"""
Model configurations for different CNN architectures and training setups.
"""

# Audio processing parameters
sample_rate = 32000
window_size = 1024
hop_size = 320
mel_bins = 64
fmin = 50
fmax = 14000

# CNN6 Configuration
cnn6_config = {
    'sample_rate': sample_rate,
    'window_size': window_size,
    'hop_size': hop_size,
    'mel_bins': mel_bins,
    'fmin': fmin,
    'fmax': fmax,
    'classes_num': 10,  # GTZAN has 10 genres
    'dropout_rate': 0.5,
    'embedding_size': 512,
}

# CNN14 Configuration
cnn14_config = {
    'sample_rate': sample_rate,
    'window_size': window_size,
    'hop_size': hop_size,
    'mel_bins': mel_bins,
    'fmin': fmin,
    'fmax': fmax,
    'classes_num': 10,  # GTZAN has 10 genres
    'dropout_rate': 0.5,
    'embedding_size': 2048,
}

# Emotion analysis configuration
emotion_config = {
    'sample_rate': sample_rate,
    'window_size': window_size,
    'hop_size': hop_size,
    'mel_bins': mel_bins,
    'fmin': fmin,
    'fmax': fmax,
    'embedding_size': 512,
    'emotion_dim': 2,  # valence and arousal
}

# Model type mappings
MODEL_CONFIGS = {
    'Transfer_Cnn6': cnn6_config,
    'Transfer_Cnn14': cnn14_config,
    'FeatureAffectiveCnn6': cnn6_config,
    'FeatureEmotionRegression_Cnn6': emotion_config,
    'EmotionRegression_Cnn6': emotion_config,
}

def get_model_config(model_type):
    """Get configuration for a specific model type."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_type] 