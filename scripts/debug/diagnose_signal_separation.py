#!/usr/bin/env python3

import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import json
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def diagnose_signal_separation():
    """Diagnose whether valence and arousal signals are properly separated."""
    
    print("🔍 STEERING SIGNAL SEPARATION DIAGNOSIS")
    print("="*60)
    
    # Load steering signals
    steering_signals_path = './steering_signals_25bin/steering_signals_25bin.json'
    if not os.path.exists(steering_signals_path):
        steering_signals_path = './tmp/25bin_steering_signals/steering_signals_25bin.json'
    
    if not os.path.exists(steering_signals_path):
        print(f"❌ Error: Steering signals file not found at {steering_signals_path}")
        return
    
    with open(steering_signals_path, 'r') as f:
        signals_25bin = json.load(f)
    
    print(f"✅ Loaded steering signals: {len(signals_25bin)} categories")
    
    # Check signal structure
    print(f"\n📊 SIGNAL STRUCTURE ANALYSIS")
    print(f"{'Category':<25} {'Has Valence':<12} {'Has Arousal':<12} {'Val Shape':<12} {'Aro Shape':<12}")
    print("-" * 80)
    
    valid_categories = []
    for category in sorted(signals_25bin.keys()):
        if category in ['metadata', 'generation_config']:
            continue
            
        signals = signals_25bin[category]
        has_valence = 'valence_128d' in signals
        has_arousal = 'arousal_128d' in signals
        
        if has_valence and has_arousal:
            val_shape = np.array(signals['valence_128d']).shape
            aro_shape = np.array(signals['arousal_128d']).shape
            valid_categories.append(category)
        else:
            val_shape = "MISSING"
            aro_shape = "MISSING"
        
        print(f"{category:<25} {has_valence:<12} {has_arousal:<12} {str(val_shape):<12} {str(aro_shape):<12}")
    
    if not valid_categories:
        print("❌ No valid categories with both valence and arousal signals!")
        return
    
    print(f"\n✅ Found {len(valid_categories)} valid categories")
    
    # Check signal similarity within each category
    print(f"\n🔍 VALENCE vs AROUSAL SIMILARITY ANALYSIS")
    print(f"{'Category':<25} {'Correlation':<12} {'Mean Diff':<12} {'Std Diff':<12} {'Same?':<8}")
    print("-" * 75)
    
    identical_count = 0
    high_correlation_count = 0
    
    for category in valid_categories:
        signals = signals_25bin[category]
        val_signal = np.array(signals['valence_128d'])
        aro_signal = np.array(signals['arousal_128d'])
        
        # Calculate correlation
        correlation = pearsonr(val_signal.flatten(), aro_signal.flatten())[0]
        
        # Calculate differences
        mean_diff = np.mean(np.abs(val_signal - aro_signal))
        std_diff = np.std(val_signal - aro_signal)
        
        # Check if signals are identical
        are_identical = np.allclose(val_signal, aro_signal, rtol=1e-5)
        
        if are_identical:
            identical_count += 1
            same_indicator = "IDENTICAL"
        elif correlation > 0.95:
            high_correlation_count += 1
            same_indicator = "TOO_SIM"
        else:
            same_indicator = "DIFF"
        
        print(f"{category:<25} {correlation:<12.4f} {mean_diff:<12.6f} {std_diff:<12.6f} {same_indicator:<8}")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   Total categories: {len(valid_categories)}")
    print(f"   Identical signals: {identical_count} ({identical_count/len(valid_categories)*100:.1f}%)")
    print(f"   High correlation (>0.95): {high_correlation_count} ({high_correlation_count/len(valid_categories)*100:.1f}%)")
    
    if identical_count > 0:
        print(f"   ❌ PROBLEM: {identical_count} categories have IDENTICAL valence and arousal signals!")
    elif high_correlation_count > len(valid_categories) * 0.5:
        print(f"   ⚠️  WARNING: {high_correlation_count} categories have very similar signals")
    else:
        print(f"   ✅ Signals appear to be properly separated")
    
    # Check signal diversity across categories
    print(f"\n🔍 SIGNAL DIVERSITY ANALYSIS")
    print(f"Checking if different categories have different signals...")
    
    # Extract all valence and arousal signals
    all_valence_signals = []
    all_arousal_signals = []
    category_names = []
    
    for category in valid_categories[:10]:  # Limit to first 10 for readability
        signals = signals_25bin[category]
        all_valence_signals.append(np.array(signals['valence_128d']))
        all_arousal_signals.append(np.array(signals['arousal_128d']))
        category_names.append(category)
    
    # Compare signals across categories
    print(f"\n{'Category A':<25} {'Category B':<25} {'Val Corr':<10} {'Aro Corr':<10}")
    print("-" * 70)
    
    high_cross_correlation = 0
    total_comparisons = 0
    
    for i in range(min(5, len(category_names))):
        for j in range(i+1, min(5, len(category_names))):
            val_corr = pearsonr(all_valence_signals[i].flatten(), all_valence_signals[j].flatten())[0]
            aro_corr = pearsonr(all_arousal_signals[i].flatten(), all_arousal_signals[j].flatten())[0]
            
            if val_corr > 0.9 or aro_corr > 0.9:
                high_cross_correlation += 1
            total_comparisons += 1
            
            print(f"{category_names[i]:<25} {category_names[j]:<25} {val_corr:<10.4f} {aro_corr:<10.4f}")
    
    print(f"\nCross-category high correlations: {high_cross_correlation}/{total_comparisons} ({high_cross_correlation/total_comparisons*100:.1f}%)")
    
    # Check expected emotion patterns
    print(f"\n🔍 EXPECTED EMOTION PATTERN ANALYSIS")
    print(f"Checking if signals match expected emotion directions...")
    
    expected_patterns = {
        'very_positive_very_strong': {'valence': 'high', 'arousal': 'high'},
        'very_negative_very_weak': {'valence': 'low', 'arousal': 'low'},
        'positive_weak': {'valence': 'high', 'arousal': 'low'},
        'negative_strong': {'valence': 'low', 'arousal': 'high'},
        'neutral_moderate': {'valence': 'mid', 'arousal': 'mid'}
    }
    
    print(f"\n{'Category':<25} {'Expected V':<10} {'Expected A':<10} {'Actual V':<10} {'Actual A':<10} {'Match':<8}")
    print("-" * 80)
    
    pattern_matches = 0
    pattern_total = 0
    
    for category, expected in expected_patterns.items():
        if category in signals_25bin:
            signals = signals_25bin[category]
            if 'valence_128d' in signals and 'arousal_128d' in signals:
                val_mean = np.mean(signals['valence_128d'])
                aro_mean = np.mean(signals['arousal_128d'])
                
                # Determine actual patterns
                val_actual = 'high' if val_mean > 0.1 else ('low' if val_mean < -0.1 else 'mid')
                aro_actual = 'high' if aro_mean > 0.1 else ('low' if aro_mean < -0.1 else 'mid')
                
                match = "✓" if (val_actual == expected['valence'] and aro_actual == expected['arousal']) else "✗"
                if match == "✓":
                    pattern_matches += 1
                pattern_total += 1
                
                print(f"{category:<25} {expected['valence']:<10} {expected['arousal']:<10} {val_actual:<10} {aro_actual:<10} {match:<8}")
    
    print(f"\nPattern matches: {pattern_matches}/{pattern_total} ({pattern_matches/pattern_total*100:.1f}%)")
    
    # Final recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if identical_count > 0:
        print(f"1. ❌ CRITICAL: Regenerate steering signals - valence and arousal are identical!")
        print(f"   → Extract signals from DIFFERENT layers or use different methodologies")
    elif high_correlation_count > len(valid_categories) * 0.5:
        print(f"2. ⚠️  WARNING: Signals are too similar between valence and arousal")
        print(f"   → Use more specialized extraction or post-processing to differentiate")
    
    if high_cross_correlation > total_comparisons * 0.7:
        print(f"3. ⚠️  WARNING: Different emotion categories have very similar signals")
        print(f"   → Improve category separation in signal generation")
    
    if pattern_matches < pattern_total * 0.5:
        print(f"4. ⚠️  WARNING: Signals don't match expected emotion patterns")
        print(f"   → Verify extraction methodology and emotion categorization")
    
    if identical_count == 0 and high_correlation_count < len(valid_categories) * 0.3 and pattern_matches > pattern_total * 0.7:
        print(f"✅ Signals appear to be properly generated and separated!")
        print(f"   → Focus on fixing application mechanism or strength tuning")

if __name__ == "__main__":
    diagnose_signal_separation() 