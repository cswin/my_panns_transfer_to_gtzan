import re
import matplotlib.pyplot as plt

# Function to extract metrics from a log file
def extract_metrics(log_path):
    iterations, valence, arousal = [], [], []
    with open(log_path, 'r') as f:
        lines = f.readlines()
    current_iter = None
    for line in lines:
        iter_match = re.search(r'Iteration: (\d+)', line)
        if iter_match:
            current_iter = int(iter_match.group(1))
        pearson_match = re.search(r'Validate Audio Valence Pearson: ([\d.]+), Arousal Pearson: ([\d.]+)', line)
        if pearson_match and current_iter is not None:
            iterations.append(current_iter)
            valence.append(float(pearson_match.group(1)))
            arousal.append(float(pearson_match.group(2)))
    return iterations, valence, arousal

# Paths to your log files
log_no_feedback = "workspaces/emotion_regression/logs/main/FeatureEmotionRegression_Cnn6_NewAffective/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/0000.log"
log_intrinsic = "workspaces/emotoin_feedback/logs/main/FeatureEmotionRegression_Cnn6_LRM/pretrain=True/loss_type=mse/augmentation=mixup/batch_size=24/freeze_base=True/0000.log"

 

# Extract data
iters_nf, val_nf, aro_nf = extract_metrics(log_no_feedback)
iters_if, val_if, aro_if = extract_metrics(log_intrinsic)

# Plot
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(iters_nf, val_nf, 'b-o', label='No Feedback Connections')
plt.plot(iters_if, val_if, 'r-o', label='Intrinsic Feedback')
plt.title('Valence')
plt.xlabel('Training Iteration')
plt.ylabel('Valence Pearson Correlation')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(iters_nf, aro_nf, 'b-o', label='No Feedback Connections')
plt.plot(iters_if, aro_if, 'r-o', label='Intrinsic Feedback')
plt.title('Arousal')
plt.xlabel('Training Iteration')
plt.ylabel('Arousal Pearson Correlation')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show() 