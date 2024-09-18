# ==========================================================================================
# gold_is_the_best_aar.py
# Description: This script is used to visualize the expected p-value vs window size for Aar.
# ==========================================================================================

import numpy as np
from scipy.stats import gamma
import matplotlib.pyplot as plt

def gamma_transform(S, W, loc=0, scale=1):
    """Calculate the expected p-value for a given window size."""
    return 1 - gamma.cdf(S, a=W, loc=loc, scale=scale)

def expected_S(W, L, mu1, mu0):
    """Calculate the expected S for a given window size."""
    if W <= L:
        return W * mu1
    else:
        return L * mu1 + (W - L) * mu0

def p_value_function(W, L, mu1, mu0):
    """Calculate the expected p-value for a given window size."""
    E_S = expected_S(W, L, mu1, mu0)
    return gamma_transform(E_S, W)

# Parameters
L = 200  # watermark length
mu1 = 1.6  # expected value of the score for the watermarked parts
mu0 = 1.0  # expected value of the score for the non-watermarked parts
alpha = 1e-6  # targeted false positive rate

# Generate a range of W values
W_range = np.arange(100, 400)

# Calculate the p-values for each W
p_values = [p_value_function(W, L, mu1, mu0) for W in W_range]

# Find the minimum p-value
min_index = np.argmin(p_values)
min_W = W_range[min_index]
min_p = p_values[min_index]

# Plot parameters
main_color = '#3498db' 
threshold_color = '#e74c3c' 
min_point_color = '#2ecc71'  
fill_color = '#85c1e9' 
background_color = '#f9f9f9' 

# Create the plot
plt.figure(figsize=(6, 4))  # 修改为与KGW一致的大小
plt.rcParams['axes.facecolor'] = background_color

# Plot the p-values
plt.plot(W_range, p_values, '-', color=main_color, linewidth=1, label='E[p-value]')
plt.axhline(y=alpha, color=threshold_color, linestyle='--', linewidth=1, label='p*')

# Fill the area where p < alpha
plt.fill_between(W_range, p_values, alpha, where=(np.array(p_values) < alpha), 
                 interpolate=True, color=fill_color, alpha=0.5)

# Plot the minimum p-value
plt.plot(min_W, min_p, 'o', color=min_point_color, markersize=10, label=f'Minimum')

# Annotate the minimum p-value
plt.xlabel('window size (W)', fontsize=10)
plt.ylabel('p-value', fontsize=10)
plt.title('p-value vs window size', fontsize=12)
plt.legend(fontsize=8)
plt.grid(True, linestyle='--', alpha=0.3)
plt.yscale('log')  # Use a logarithmic scale for the y-axis
plt.tick_params(axis='y', which='both', left=False, right=False)
plt.tight_layout()

plt.savefig('fig/gold_is_the_best_aar.png', dpi=300, bbox_inches='tight')