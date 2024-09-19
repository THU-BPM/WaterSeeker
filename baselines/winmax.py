# ==================================================================
# winmax.py
# Description: WinMax baseline for watermarked segments detection.
# ==================================================================

import os
import json
import torch
import argparse
from tqdm import tqdm
from watermark.kgw import KGW
from watermark.aar import AAR
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for watermarking')
    parser.add_argument('--watermark', type=str, default='kgw')
    parser.add_argument('--input_file', type=str, default='data/non_watermarked_data.json')
    parser.add_argument('--min_window_length', type=int, default=100)
    parser.add_argument('--max_window_length', type=int, default=400)
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', type=str, default='baseline_result/kgw_winmax.log',
                        help='Output file path')
    parser.add_argument('--model', type=str, default='llama')
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
    with open(args.input_file) as f:
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

            detect_result = watermark.detect_watermark_win_max(text=text, min_L=args.min_window_length, max_L=args.max_window_length)
            is_watermarked = detect_result['is_watermarked']
            indices = detect_result['indices']

            f.write(json.dumps({'predicted': is_watermarked, 'indices': indices, 'gold': flag, 'gold_indices': (start_index, end_index)}) + '\n')
    
