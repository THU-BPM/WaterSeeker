import json
import argparse
import numpy as np
from tqdm import tqdm
from scipy.stats import binom, norm

def P_z_greater_equal_z_star(z_star, T, p):
    """Calculate P(z >= z*)."""
    prob_list = []
    for k in range(int(np.ceil(p*T + z_star * np.sqrt(T*p*(1-p)))), T+1):
        prob_list.append(binom.pmf(k, T, p))
    
    P = sum(prob_list)
    
    return P

def find_z_star(T, alpha, p=0.5):
    """Find the minimum z_star such that P(z >= z*) <= alpha."""
    z_star = 0
    while True:
        P_z_greater_equal_z_star_value = P_z_greater_equal_z_star(z_star, T, p)
        if P_z_greater_equal_z_star_value <= alpha:
            return z_star
        z_star += 0.01

def z(alpha):
    return norm.ppf(1 - alpha)


def compute_z_threshold(T, alpha, p=0.5):
    """计算z_threshold，从T=1到T=1000，当T<200时用find_z_star，当T>=200时用z(alpha)"""
    if T < 200:
        z_threshold = find_z_star(T, alpha, p)
    else:
        z_threshold = z(alpha)
    return z_threshold

if __name__ == '__main__':
    # 从T=1到T=1000，计算z_threshold，并保存成字典，用argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=5e-7)
    parser.add_argument('--p', type=float, default=0.5)
    parser.add_argument('--T_min', type=int, default=1)
    parser.add_argument('--T_max', type=int, default=1000)
    parser.add_argument('--output_file', type=str, default='threshold_dict/z_threshold_dict_5e7.json')
    args = parser.parse_args()

    z_threshold_dict = {}
    for T in tqdm(range(args.T_min, args.T_max + 1)):
        z_threshold_dict[T] = compute_z_threshold(T, args.alpha, args.p)
    with open(args.output_file, 'w') as f:
        json.dump(z_threshold_dict, f)
