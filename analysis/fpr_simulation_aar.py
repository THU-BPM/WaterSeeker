# ==============================================================================
# fpr_simulation_aar.py
# Description: This script is used to simulate the document-level FPR given the 
#              targeted FPR within the detection window for Aar.
# ==============================================================================

import os
import json
import scipy
import numpy as np
from math import log
from tqdm import tqdm

def generate_simulation_data_aar(num_samples=10000, length=10000, output_file='data/aar/threshold_validation/validation.json'):
    """Generate simulation data for Aar."""
    # For each sample, generate a sequence of length, where each element in the sequence is sampled from a uniform distribution in [0, 1]
    data = []
    for i in tqdm(range(num_samples)):
        sample = np.random.uniform(0, 1, length)
        data.append(sample.tolist())

    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for sample in data:
            f.write(json.dumps(sample) + '\n')

def detect_aar(input_file='data/aar/threshold_validation/validation.json', window_size=50, 
               threshold_1=0.5, threshold_2=1.5, top_k=20, min_length=100, tolerance=100):
    """Detect Aar watermark."""
    # Read file
    with open(input_file, 'r') as f:
        data = f.readlines()
    
    fp = 0

    for sample in tqdm(data):
        u_list = json.loads(sample)
        token_scores = []
        for i in range(len(u_list)):
            token_scores.append(log(1 / (1 - u_list[i])))
         
        # Detect anomalies
        # Compute average of token scores in each window
        proportions = []
        for i in range(len(token_scores) - window_size + 1):
            window = token_scores[i:i + window_size]
            proportion = sum(window) / window_size
            proportions.append(proportion)

        # Find significantly higher windows
        mean_proportion = np.mean(proportions)
        std_proportion = np.std(proportions)

        # Compute the mean of the top-k proportions
        top_proportions = sorted(proportions, reverse=True)[:top_k]
        top_mean_proportion = np.mean(top_proportions)

        diff_value = max((top_mean_proportion - mean_proportion) * threshold_1, std_proportion * threshold_2)

        anomalies = [i for i, p in enumerate(proportions) if p > mean_proportion + diff_value]

        # Merge adjacent or nearly adjacent anomaly segments
        merged_anomalies = []
        current_segment = []

        for i in range(len(anomalies)):
            if not current_segment:
                current_segment = [anomalies[i]]
            else:
                if anomalies[i] - current_segment[-1] <= tolerance:
                    current_segment.append(anomalies[i])
                else:
                    merged_anomalies.append(current_segment)
                    current_segment = [anomalies[i]]
        
        if current_segment:
            merged_anomalies.append(current_segment)

        valid_segments = []

        # Filter out segments that are too short, which are considered noise
        for segment in merged_anomalies:
            if (min_length <= (segment[-1] - segment[0] + window_size - 1)):
                valid_segments.append((segment[0], segment[-1] + window_size - 1))

        indices = valid_segments

        if not indices:
            result = (False, [])
        else:
            is_watermarked = False
            filtered_indices = []

            for indice in indices:
                found_in_current_indice = False  # Flag variable, used to mark whether the current indice has found a segment that meets the conditions
                
                min_p = float('inf')
                min_index = ()
                for start_idx in range(indice[0], indice[0] + window_size):
                    for end_idx in range(indice[-1], indice[-1] - window_size, -1):
                        if end_idx - start_idx < min_length:
                            break
                        p_value = scipy.stats.gamma.sf(sum(token_scores[start_idx:end_idx]), end_idx - start_idx, loc=0, scale=1)
                        if p_value < 1e-6:
                            is_watermarked = True
                            if p_value < min_p:
                                min_p = p_value
                                min_index = (start_idx, end_idx, p_value)
                            found_in_current_indice = True  # Found a segment that meets the conditions
                if found_in_current_indice:
                    filtered_indices.append(min_index)

            result = (is_watermarked, filtered_indices)
        
        if result[0]:
            fp += 1
    
    fpr = fp / len(data)
    print(f'FPR: {fpr}')

if __name__ == '__main__':
    generate_simulation_data_aar(10000, 10000, 'data/aar/fpr_simulation/data.json')
    detect_aar('data/aar/fpr_simulation/data.json', window_size=50, threshold_1=0.5, 
           threshold_2=1.5, top_k=20, min_length=100, tolerance=100)
