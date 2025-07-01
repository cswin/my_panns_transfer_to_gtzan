#!/usr/bin/env python3
"""
Debug the actual signal values to see if 9-bin and 25-bin are using identical signals.
"""

import os
import sys
import torch
import numpy as np
import json

def debug_signal_values():
    """Compare the actual signal values between 9-bin and 25-bin methods."""
    
    # Load steering signals
    with open('tmp/steering_signals_by_category.json', 'r') as f:
        signals_9bin = json.load(f)
    with open('tmp/25bin_steering_signals/steering_signals_25bin.json', 'r') as f:
        signals_25bin = json.load(f)
    
    # Compare the categories we're using
    category_9bin = 'negative_strong'
    category_25bin = 'very_negative_very_strong'
    
    print(f"🔍 COMPARING SIGNAL VALUES:")
    print(f"9-bin category: {category_9bin}")
    print(f"25-bin category: {category_25bin}")
    
    if category_9bin in signals_9bin and category_25bin in signals_25bin:
        signals_9 = signals_9bin[category_9bin]
        signals_25 = signals_25bin[category_25bin]
        
        print(f"\n📊 VALENCE SIGNALS:")
        if 'valence_128d' in signals_9 and 'valence_128d' in signals_25:
            val_9 = np.array(signals_9['valence_128d'])
            val_25 = np.array(signals_25['valence_128d'])
            
            print(f"9-bin valence shape: {val_9.shape}")
            print(f"25-bin valence shape: {val_25.shape}")
            print(f"9-bin valence mean: {val_9.mean():.6f}")
            print(f"25-bin valence mean: {val_25.mean():.6f}")
            print(f"9-bin valence std: {val_9.std():.6f}")
            print(f"25-bin valence std: {val_25.std():.6f}")
            
            # Check if they're identical
            if np.allclose(val_9, val_25):
                print(f"⚠️  VALENCE SIGNALS ARE IDENTICAL!")
            else:
                diff = np.abs(val_9 - val_25)
                print(f"✅ Valence signals differ - max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")
            
            # Show first few values
            print(f"9-bin valence[:5]: {val_9[:5]}")
            print(f"25-bin valence[:5]: {val_25[:5]}")
        
        print(f"\n📊 AROUSAL SIGNALS:")
        if 'arousal_128d' in signals_9 and 'arousal_128d' in signals_25:
            ar_9 = np.array(signals_9['arousal_128d'])
            ar_25 = np.array(signals_25['arousal_128d'])
            
            print(f"9-bin arousal shape: {ar_9.shape}")
            print(f"25-bin arousal shape: {ar_25.shape}")
            print(f"9-bin arousal mean: {ar_9.mean():.6f}")
            print(f"25-bin arousal mean: {ar_25.mean():.6f}")
            print(f"9-bin arousal std: {ar_9.std():.6f}")
            print(f"25-bin arousal std: {ar_25.std():.6f}")
            
            # Check if they're identical
            if np.allclose(ar_9, ar_25):
                print(f"⚠️  AROUSAL SIGNALS ARE IDENTICAL!")
            else:
                diff = np.abs(ar_9 - ar_25)
                print(f"✅ Arousal signals differ - max diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")
            
            # Show first few values
            print(f"9-bin arousal[:5]: {ar_9[:5]}")
            print(f"25-bin arousal[:5]: {ar_25[:5]}")
    
    # Let's also check what categories are available
    print(f"\n📊 AVAILABLE CATEGORIES:")
    print(f"9-bin categories: {list(signals_9bin.keys())}")
    print(f"25-bin categories: {list(signals_25bin.keys())}")
    
    # Check if there are truly different categories we should be comparing
    print(f"\n🔍 SUGGESTED BETTER COMPARISON:")
    
    # Find a 25-bin category that should be clearly different
    better_25bin_categories = [
        'neutral_middle',      # Should be very different from negative_strong
        'positive_strong',     # Opposite valence
        'very_negative_weak',  # Same valence but different arousal
    ]
    
    for alt_category in better_25bin_categories:
        if alt_category in signals_25bin:
            print(f"✅ Could compare 9-bin '{category_9bin}' vs 25-bin '{alt_category}'")
            
            signals_alt = signals_25bin[alt_category]
            if 'valence_128d' in signals_9 and 'valence_128d' in signals_alt:
                val_9 = np.array(signals_9['valence_128d'])
                val_alt = np.array(signals_alt['valence_128d'])
                
                diff = np.abs(val_9 - val_alt)
                print(f"   Valence difference: max={diff.max():.6f}, mean={diff.mean():.6f}")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"If signals are identical, that explains why outputs are identical!")
    print(f"We need to test with genuinely different emotion categories.")

if __name__ == "__main__":
    debug_signal_values() 