import json
import numpy as np
import matplotlib.pyplot as plt

# Load data
with open('workspaces/reproducible_steering_results.json', 'r') as f:
    data = json.load(f)

strengths = np.array(data['strengths'])
valence_corr = np.array(data['valence_correlations'])
arousal_corr = np.array(data['arousal_correlations'])
baseline_valence = data['baseline_valence']
baseline_arousal = data['baseline_arousal']
best_valence_strength = data['best_valence_strength']
best_arousal_strength = data['best_arousal_strength']

# Bar width and positions for grouped bars
bar_width = 0.4
x = np.arange(len(strengths))

fig, ax = plt.subplots(figsize=(12, 7))

# Bar plots
def highlight_bar_indices(strengths, best_strength):
    return [i for i, s in enumerate(strengths) if s == best_strength]

valence_bars = ax.bar(x - bar_width/2, valence_corr, width=bar_width, label='Valence', color='skyblue', edgecolor='black')
arousal_bars = ax.bar(x + bar_width/2, arousal_corr, width=bar_width, label='Arousal', color='orchid', edgecolor='black')

# Highlight peak bars
for idx in highlight_bar_indices(strengths, best_valence_strength):
    valence_bars[idx].set_edgecolor('gold')
    valence_bars[idx].set_linewidth(3)
for idx in highlight_bar_indices(strengths, best_arousal_strength):
    arousal_bars[idx].set_edgecolor('gold')
    arousal_bars[idx].set_linewidth(3)

# Trend lines (moving average for smoothing)
def moving_average(y, window=3):
    return np.convolve(y, np.ones(window)/window, mode='same')

ax.plot(x, moving_average(valence_corr), color='blue', linestyle='--', label='Valence Trend')
ax.plot(x, moving_average(arousal_corr), color='purple', linestyle='--', label='Arousal Trend')

# Baseline lines
ax.axhline(baseline_valence, color='skyblue', linestyle=':', label=f'Baseline Valence (r={baseline_valence:.3f})')
ax.axhline(baseline_arousal, color='orchid', linestyle=':', label=f'Baseline Arousal (r={baseline_arousal:.3f})')

# X-axis: log scale labels
ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in strengths], rotation=45)
ax.set_xlabel('Steering Strength')
ax.set_ylabel('Pearson Correlation')
ax.set_title('25-Bin Steering Signal Performance vs Strength (Bar Plot)')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show() 