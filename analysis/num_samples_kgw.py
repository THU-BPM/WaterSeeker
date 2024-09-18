# =============================================================================
# num_samples_kgw.py
# Description: This script is used to visualize the sample distribution for KGW.
# =============================================================================

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set delta and length ranges
deltas = [2.0, 1.5, 1.0]
length_ranges = ['[100, 200)', '[200, 300)', '[300, 400)']

# Define the number of samples
samples = np.array([[43, 34, 37], [38, 32, 32], [38, 20, 26]]) # Llama
samples = np.array([[37, 19, 44], [39, 31, 32], [33, 34, 31]]) # Mistral

# Create the plot
plt.figure(figsize=(5, 4))

sns.heatmap(samples, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=length_ranges, yticklabels=deltas,
            annot_kws={"size": 14})

plt.xlabel('Length Range')
plt.ylabel('Delta')
plt.savefig('fig/num_samples_kgw_llama.png')