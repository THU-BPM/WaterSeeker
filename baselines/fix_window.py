# ==============================================================
# fix_window.py
# Description: FLSW baseline for watermarked segments detection.
# ==============================================================

import os
import json
import torch
import argparse
from tqdm import tqdm
from watermark.kgw import KGW
from watermark.aar import AAR
from utils.transformers_config import TransformersConfig
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

def find_p_star(alpha):
    """Find the p* value for a given alpha."""
    return alpha

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FLSW baseline for watermark detection.')
    parser.add_argument('--watermark', type=str, default='kgw')
    parser.add_argument('--input_file', type=str, default='data/non_watermarked_data.json')
    parser.add_argument('--targeted_fpr', type=float, default=1e-6)
    parser.add_argument('--z_threshold_dict', type=str, default='threshold_dict/z_threshold_dict_1e6.json')
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--output_file', type=str, default='kgw_fix_window_100_fpr.json',
                        help='Output file path')
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', type=str, default='llama')

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    if args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained('/workspace/intern_ckpt/panleyi/Llama-2-7b-hf')

        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained('/workspace/intern_ckpt/panleyi/Llama-2-7b-hf', device_map='auto'),
                                            tokenizer=tokenizer,
                                            vocab_size=32000,
                                            device=device)

    elif args.model == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained('/workspace/intern_ckpt/panleyi/Mistral-7B-v0.1')

        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained('/workspace/intern_ckpt/panleyi/Mistral-7B-v0.1', device_map='auto'),
                                                tokenizer=tokenizer,
                                                vocab_size=32000,
                                                device=device)
        
    # Load z-threshold dictionary
    with open(args.z_threshold_dict, 'r') as f:
        z_threshold_dict = json.load(f)
    
    # Load watermark
    if args.watermark == 'kgw':
        watermark = KGW('config/KGW.json', transformers_config=transformers_config)
        threshold = z_threshold_dict[str(args.window_size)]
        watermark.config.hash_key = args.key
        
    elif args.watermark == 'aar':
        watermark = AAR('config/AAR.json', transformers_config=transformers_config)
        threshold = find_p_star(args.targeted_fpr)
        watermark.config.seed = args.seed

    # Read file
    with open(args.input_file, 'r') as f:
        data = f.readlines()
    
    # Detect watermark
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for d in tqdm(data):
            loaded_data = json.loads(d)
            text = loaded_data['text']
            flag = loaded_data['flag']
            segments = loaded_data['segments']

            detect_result = watermark.detect_watermark_with_fix_window(text=text, L=args.window_size, threshold=threshold)
            is_watermarked = detect_result['is_watermarked']
            indices = detect_result['indices']

            f.write(json.dumps({'predicted': is_watermarked, 'indices': indices, 'gold': flag, 'gold_indices': segments}) + '\n')

        
           

