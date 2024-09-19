# ==========================================
# generate_data.py
# Description: Generated watermarked data.
# ==========================================

import os
import json
import torch
import random
import argparse
from tqdm import tqdm
from watermark.kgw import KGW
from watermark.aar import AAR
from utils.transformers_config import TransformersConfig
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for watermarking')
    parser.add_argument('--num_sample', type=int, default=300)
    parser.add_argument('--wiki_length', type=int, default=10000)
    parser.add_argument('--wiki_start_index', type=int, default=0)
    parser.add_argument('--prompt_start_index', type=int, default=0)
    parser.add_argument('--min_length', type=int, default=100)
    parser.add_argument('--max_length', type=int, default=400)
    parser.add_argument('--watermark', type=str, default='kgw')
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', type=str, default='data/kgw/llama/data_delta_1.json',
                        help='Output file path')
    parser.add_argument('--model', type=str, default='llama')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained('/data2/shared_model/llama-2-7b-hf')

        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained('/data2/shared_model/llama-2-7b-hf', device_map='auto'),
                                            tokenizer=tokenizer,
                                            vocab_size=32000,
                                            device=device,
                                            max_new_tokens=args.max_length,
                                            min_length=args.max_length + 30,
                                            do_sample=True)

    elif args.model == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained('/data2/shared_model/mistral-7b-v0.1')

        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained('/data2/shared_model/mistral-7b-v0.1', device_map='auto'),
                                                tokenizer=tokenizer,
                                                vocab_size=32000,
                                                device=device,
                                                max_new_tokens=args.max_length,
                                                min_length=args.max_length + 30,
                                                do_sample=True)

    if args.watermark == 'kgw':
        watermark = KGW('config/KGW.json', transformers_config=transformers_config)     
        watermark.config.hash_key = args.key

    elif args.watermark == 'aar':
        watermark = AAR('config/AAR.json', transformers_config=transformers_config)
        watermark.config.seed = args.seed

    with open('data/c4/processed_c4.json', 'r') as f:
        prompt_data = f.readlines()
    
    with open('data/wikipedia/long_document.json', 'r') as f:
        wiki_data = json.load(f)
    
    data = []

    index = 0

    # generate watermarked data
    for d in tqdm(prompt_data[args.prompt_start_index: args.prompt_start_index + args.num_sample]):
        prompt = json.loads(d)['prompt']

        # random sample watermark strength
        if args.watermark == 'kgw':
            watermark.config.delta = random.choice([2.0, 1.5, 1.0])
        elif args.watermark == 'aar':
            watermark.config.temperature = random.choice([0.3, 0.2, 0.1])
        
        # generate watermarked text
        watermarked_text = watermark.generate_watermarked_text(prompt)
        watermarked_tokens = tokenizer.encode(watermarked_text, return_tensors='pt', add_special_tokens=False).squeeze()
        
        # truncate prompt from watermarked tokens
        prompt_tokens = tokenizer.encode(prompt, return_tensors='pt', add_special_tokens=False).squeeze()
        watermarked_tokens = watermarked_tokens[len(prompt_tokens):]
        
        # truncate watermarked tokens to a random length from min_length to max_length
        random_length = torch.randint(args.min_length, args.max_length + 1, (1,)).item()
        watermarked_tokens_fragment = watermarked_tokens[:random_length]

        # insert watermarked text into a random position in the wiki text and record start_index and end_index
        tokens = tokenizer.encode(wiki_data[index + args.wiki_start_index]['text'], return_tensors='pt', add_special_tokens=False).squeeze()[:args.wiki_length]
        insert_position = torch.randint(0, len(tokens), (1,)).item()
        tokens = torch.cat((tokens[:insert_position], watermarked_tokens_fragment, tokens[insert_position:]))
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        data.append({'text': text, 'flag': 1, 'start_index': insert_position, 'end_index': insert_position+random_length})
        index += 1
    
    # save data
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for d in data:
            f.write(json.dumps(d)+'\n')
