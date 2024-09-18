# ==============================================================================
# fpr_simulation_kgw.py
# Description: This script is used to simulate the document-level FPR given the 
#              targeted FPR within the detection window for KGW.
# ==============================================================================

import os
import json
import numpy as np
from tqdm import tqdm
from math import sqrt

def generate_simulation_data_kgw(num_samples=10000, length=10000, output_file='data/kgw/threshold_validation/validation.json'):
    """Generate simulation data for KGW."""
    # For each sample, generate a sequence of length, where each element in the sequence is sampled from a Bernoulli distribution with p=0.5
    data = []
    for i in tqdm(range(num_samples)):
        sample = np.random.choice([0, 1], length, p=[0.5, 0.5])
        data.append(sample.tolist())

    # Write to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for sample in data:
            f.write(json.dumps(sample) + '\n')

def score_sequence_by_window(token_flags: list, start_idx: int, end_idx: int) -> float:
    """Compute z-score for a window of length L starting at start_idx."""
    green_token_count = sum(token_flags[start_idx: end_idx])
    expected_count = 0.5
    T = end_idx - start_idx
    numer = green_token_count - expected_count * T 
    denom = sqrt(T * expected_count * (1 - expected_count))  
    z_score = numer / denom
    return z_score

def detect_kgw(input_file='data/kgw/threshold_validation/validation.json', window_size=50, 
               threshold_1=0.5, threshold_2=1.5, top_k=20, min_length=100, tolerance=100):
    """Detect KGW watermark."""
    # Load z-threshold dictionary
    with open('threshold_dict/z_threshold_dict_1e6.json', 'r') as f:
        z_threshold_dict = json.load(f)

    # Read file
    with open(input_file, 'r') as f:
        data = f.readlines()
    
    fp = 0

    for sample in tqdm(data):
        token_flags = json.loads(sample)
        
        # Detect anomalies
        # Compute the proportion of green tokens in each window
        proportions = []
        for i in range(len(token_flags) - window_size + 1):
            window = token_flags[i:i + window_size]
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

                max_z = -100
                max_index = ()
                
                for start_idx in range(indice[0], indice[0] + window_size):
                    for end_idx in range(indice[-1], indice[-1] - window_size, -1):
                        if end_idx - start_idx < min_length:
                            break
                        
                        if end_idx - start_idx > 1000:
                            z_threshold = z_threshold_dict['1000']
                        else:
                            z_threshold = z_threshold_dict[str(end_idx - start_idx)]
                        z_score = score_sequence_by_window(token_flags, start_idx, end_idx)
                        if z_score > z_threshold:
                            is_watermarked = True
                            if z_score > max_z:
                                max_z = z_score
                                max_index = (start_idx, end_idx, z_score)
                            found_in_current_indice = True  # Found a segment that meets the conditions
                if found_in_current_indice:
                    filtered_indices.append(max_index)
                        
            result = (is_watermarked, filtered_indices)
        
        if result[0]:
            fp += 1
    
    fpr = fp / len(data)
    print(f'FPR: {fpr}')


if __name__ == '__main__':
    generate_simulation_data_kgw(10000, 10000, 'data/kgw/fpr_simulation/data.json')
    detect_kgw('data/kgw/fpr_simulation/data.json', window_size=50, threshold_1=0.5, 
            threshold_2=1.5, top_k=20, min_length=100, tolerance=100)
