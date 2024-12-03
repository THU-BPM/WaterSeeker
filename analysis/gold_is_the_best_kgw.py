# =========================================================================================
# gold_is_the_best_kgw.py
# Description This script is used to visualize the expected z-score vs window size for KGW.
# =========================================================================================

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def z_score(W, L, gamma, gamma1):
    """Calculate the expected z-score for a given window size."""
    if W <= L:
        return np.sqrt(W) * (gamma1 - gamma) / np.sqrt(gamma * (1 - gamma))
    else:
        return L * (gamma1 - gamma) / np.sqrt(gamma * (1 - gamma) * W)

# Parameters
L = 200  # watermark length
gamma1 = 0.75  # expected value of the score for the watermarked parts
gamma = 0.5  # expected value of the score for the non-watermarked parts
alpha = 1e-6  # targeted false positive rate

# Generate a range of W values
W_range = np.arange(100, 400)

# Calculate the z-threshold under the targeted false positive rate
z_star = norm.ppf(1 - alpha)

# Calculate the z-scores for each W
z_scores = [z_score(W, L, gamma, gamma1) for W in W_range]

# Plot parameters
main_color = '#3498db'
threshold_color = '#e74c3c'
fill_color = '#85c1e9'
peak_color = '#2ecc71'
background_color = '#f9f9f9'

# 设置Times New Roman字体
plt.rcParams['font.family'] = 'Times New Roman'

# Create the plot
plt.figure(figsize=(4, 3))
plt.rcParams['axes.facecolor'] = background_color

# Plot the z-scores
plt.plot(W_range, [z_star] * len(W_range), '--', color=threshold_color, linewidth=1, label='z*')
plt.plot(W_range, z_scores, '-', color=main_color, linewidth=1, label='E[z-score]')

# Fill the area where z > z*
plt.fill_between(W_range, z_star, z_scores, where=(np.array(z_scores) > z_star), 
                 interpolate=True, color=fill_color, alpha=0.5)

# Find the peak z-score
peak_index = np.argmax(z_scores)
plt.plot(W_range[peak_index], z_scores[peak_index], 'o', color=peak_color, markersize=9, label='Peak')

# Set the labels and title
plt.xlabel('window size (W)', fontsize=14)
plt.ylabel('z-score', fontsize=14)
plt.legend(fontsize=14, prop={'weight': 'bold'})
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

plt.savefig('fig/gold_is_the_best_kgw.png', dpi=300, bbox_inches='tight')