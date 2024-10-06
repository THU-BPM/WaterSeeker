# ================================================================
# seeker.py
# Description: Implementation of our proposed method WaterSeeker.
# ================================================================

import os
import json
import torch
import argparse
from tqdm import tqdm
from watermark.kgw import KGW
from watermark.aar import AAR
from utils.transformers_config import TransformersConfig
from transformers import LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for watermarking')
    parser.add_argument('--watermark', type=str, default='kgw')
    parser.add_argument('--targeted_fpr', type=float, default=1e-6)
    parser.add_argument('--input_file', type=str, default='data/kgw/main/data.json')
    parser.add_argument('--output_file', type=str, default='kgw_seeker_1_output.log',
                        help='Output file path')
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--threshold_1', type=float, default=0.5)
    parser.add_argument('--threshold_2', type=float, default=1.5)
    parser.add_argument('--min_length', type=int, default=100)
    parser.add_argument('--model', type=str, default='llama')
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    if args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained('/data2/shared_model/llama-2-7b-hf')

        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained('/data2/shared_model/llama-2-7b-hf', device_map='auto'),
                                            tokenizer=tokenizer,
                                            vocab_size=32000,
                                            device=device)

    elif args.model == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained('/data2/shared_model/mistral-7b-v0.1')

        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained('/data2/shared_model/mistral-7b-v0.1', device_map='auto'),
                                                tokenizer=tokenizer,
                                                vocab_size=32000,
                                                device=device)
    
    # Load watermark
    if args.watermark == 'kgw':
        watermark = KGW('config/KGW.json', transformers_config=transformers_config)
        watermark.config.hash_key = args.key
    elif args.watermark == 'aar':
        watermark = AAR('config/AAR.json', transformers_config=transformers_config)
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
            start_index = loaded_data['start_index']
            end_index = loaded_data['end_index']

            detect_result = watermark.detect_watermark_with_seeker(text=text,
                                                                   targeted_fpr=args.targeted_fpr, 
                                                                   window_size=args.window_size, 
                                                                   threshold_1=args.threshold_1,
                                                                   threshold_2=args.threshold_2,
                                                                   top_k=20,
                                                                   min_length=args.min_length,
                                                                   tolerance=100)
            is_watermarked = detect_result['is_watermarked']
            indices = detect_result['indices']

            f.write(json.dumps({'predicted': is_watermarked, 'indices': indices, 'gold': flag, 'gold_indices': (start_index, end_index)}) + '\n')
            

        
           


