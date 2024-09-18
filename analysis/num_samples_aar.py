# =============================================================================
# num_samples_aar.py
# Description: This script is used to visualize the sample distribution for Aar.
# =============================================================================

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set temperature and length ranges
temps = [0.3, 0.2, 0.1]
length_ranges = ['[100, 200)', '[200, 300)', '[300, 400)']

# Define the number of samples
samples = np.array([[33, 25, 32], [48, 32, 22], [32, 39, 37]]) # Llama
samples = np.array([[47, 31, 41], [43, 40, 19], [32, 21, 26]]) # Mistral

# Create the plot
plt.figure(figsize=(5, 4))

sns.heatmap(samples, annot=True, fmt='d', cmap='YlGnBu', 
            xticklabels=length_ranges, yticklabels=temps,
            annot_kws={"size": 14})

plt.xlabel('Length Range')
plt.ylabel('Temperature')
plt.savefig('fig/num_samples_aar_llama.png')