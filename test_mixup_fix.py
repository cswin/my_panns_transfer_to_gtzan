#!/usr/bin/env python3

import torch
import numpy as np
import sys
import os

# Add pytorch and utils directories to path
sys.path.append('pytorch')
sys.path.append('utils')

from pytorch.pytorch_utils import do_mixup
from utils.utilities import Mixup

def test_mixup_fix():
    """Test that the mixup fix works with CUDA tensors."""
    
    print("Testing Mixup Fix...")
    
    # Test parameters - note: do_mixup expects batch_size*2 input, mixup.get_lambda expects batch_size
    batch_size = 4
    feature_shape = (batch_size * 2, 64, 64)  # (8, time, mel_bins) - input for do_mixup
    
    # Create test data
    x = torch.randn(feature_shape)
    
    # Test with CPU tensors first
    print("Testing with CPU tensors...")
    mixup = Mixup(mixup_alpha=1.0)
    # The mixup.get_lambda should return batch_size*2 values for the batch_size*2 input
    mixup_lambda = mixup.get_lambda(batch_size * 2)
    
    try:
        result_cpu = do_mixup(x, mixup_lambda)
        print(f"✅ CPU mixup successful: {x.shape} -> {result_cpu.shape}")
    except Exception as e:
        print(f"❌ CPU mixup failed: {e}")
        return False
    
    # Test with CUDA tensors if available
    if torch.cuda.is_available():
        print("Testing with CUDA tensors...")
        x_cuda = x.cuda()
        
        try:
            result_cuda = do_mixup(x_cuda, mixup_lambda)
            print(f"✅ CUDA mixup successful: {x_cuda.shape} -> {result_cuda.shape}")
            print(f"   Result device: {result_cuda.device}")
        except Exception as e:
            print(f"❌ CUDA mixup failed: {e}")
            return False
    else:
        print("CUDA not available, skipping CUDA test")
    
    print("All mixup tests passed!")
    return True

if __name__ == '__main__':
    success = test_mixup_fix()
    sys.exit(0 if success else 1) 