import json
import matplotlib.pyplot as plt
import numpy as np

# Load results from JSON
json_path = 'steering_test_results/steering_results.json'
with open(json_path, 'r') as f:
    results = json.load(f)

# Extract data
strengths = []
val_corrs = []
aro_corrs = []
for k, v in results.items():
    strengths.append(float(k))
    val_corrs.append(float(v['val_corr']))
    aro_corrs.append(float(v['aro_corr']))

# Sort by strength
strengths, val_corrs, aro_corrs = zip(*sorted(zip(strengths, val_corrs, aro_corrs)))
strengths = np.array(strengths)
val_corrs = np.array(val_corrs)
aro_corrs = np.array(aro_corrs)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(strengths, val_corrs, 'o-', label='Valence r', color='#2E86AB')
plt.plot(strengths, aro_corrs, 's-', label='Arousal r', color='#A23B72')

# Find and annotate maximum values
val_max_idx = np.argmax(val_corrs)
aro_max_idx = np.argmax(aro_corrs)

plt.annotate(f'Max: {val_corrs[val_max_idx]:.3f}', 
            xy=(strengths[val_max_idx], val_corrs[val_max_idx]),
            xytext=(0, -30), textcoords='offset points',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#2E86AB', alpha=0.7),
            fontsize=10, color='white')

plt.annotate(f'Max: {aro_corrs[aro_max_idx]:.3f}', 
            xy=(strengths[aro_max_idx], aro_corrs[aro_max_idx]),
            xytext=(0, -30), textcoords='offset points',
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#A23B72', alpha=0.7),
            fontsize=10, color='white')

plt.xlabel('Steering Strength', fontsize=13)
plt.ylabel('Pearson Correlation (r)', fontsize=13)
plt.title('Valence & Arousal Performance vs Steering Strength', fontsize=15)
plt.xticks(strengths, rotation=45)  # Show all strength values on x-axis
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.savefig('workspaces/steering_performance_vs_strength.png', dpi=200)
plt.show() 