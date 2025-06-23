"""
Training and evaluation modules for music genre classification models.
"""

from src.training.trainer import *
from src.training.evaluator import *
from src.training.evaluator_lrm import *
from src.training.config import *

__all__ = [
    'train',
    'evaluate',
    'evaluate_lrm',
    'sample_rate',
    'cnn6_config',
    'cnn14_config',
] 