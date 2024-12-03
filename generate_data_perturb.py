import os
import json
import torch
import random
import argparse
from tqdm import tqdm
import concurrent.futures
from watermark.kgw import KGW
from watermark.aar import AAR
from utils.transformers_config import TransformersConfig
from utils.utils import GPTParaphraser, WordDeletion, SynonymSubstitution
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM

def generate_data_for_gpu(rank, args, world_size):
    device = f"cuda:{rank}"
    
    if args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained('/workspace/intern_ckpt/panleyi/Llama-2-7b-hf')
        transformers_config = TransformersConfig(
            model=AutoModelForCausalLM.from_pretrained('/workspace/intern_ckpt/panleyi/Llama-2-7b-hf', device_map=device),
            tokenizer=tokenizer,
            vocab_size=32000,
            device=device,
            max_new_tokens=args.max_length,
            min_length=args.max_length + 30,
            do_sample=True,
            no_repeat_ngram_size=4
        )
    elif args.model == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained('/workspace/intern_ckpt/panleyi/Mistral-7B-v0.1')
        transformers_config = TransformersConfig(
            model=AutoModelForCausalLM.from_pretrained('/workspace/intern_ckpt/panleyi/Mistral-7B-v0.1', device_map=device),
            tokenizer=tokenizer,
            vocab_size=32000,
            device=device,
            max_new_tokens=args.max_length,
            min_length=args.max_length + 30,
            do_sample=True,
            no_repeat_ngram_size=4
        )

    if args.watermark == 'kgw':
        watermark = KGW('config/KGW.json', transformers_config=transformers_config)     
        watermark.config.hash_key = args.key
    elif args.watermark == 'aar':
        watermark = AAR('config/AAR.json', transformers_config=transformers_config)
        watermark.config.seed = args.seed

    if args.attack_type == 'paraphrase':
        attacker = GPTParaphraser(openai_model='gpt-3.5-turbo',
                                 prompt='Please rewrite the following text: ')
    elif args.attack_type == 'deletion':
        attacker = WordDeletion(ratio=0.3)
    elif args.attack_type == 'substitution':
        attacker = SynonymSubstitution(ratio=0.3)

    with open('data/c4/processed_c4.json', 'r') as f:
        prompt_data = f.readlines()
    
    with open('data/wikipedia/long_document.json', 'r') as f:
        wiki_data = json.load(f)
    
    data = []
    para_data = []
    
    # Start from rank to ensure different GPUs start with different prompts
    prompt_idx = rank
    wiki_idx = rank
    
    pbar = tqdm(total=args.num_sample // world_size, desc=f'GPU {rank}')
    
    while True:
        # Check shared counter file to see if we've reached total samples
        counter_file = os.path.join(os.path.dirname(args.output_file), 'sample_counter.txt')
        try:
            with open(counter_file, 'r') as f:
                total_samples = int(f.read().strip())
            if total_samples >= args.num_sample:
                break
        except FileNotFoundError:
            total_samples = 0

        if prompt_idx >= len(prompt_data):
            prompt_idx = rank  # Reset to initial offset if we run out of prompts
        
        prompt = json.loads(prompt_data[prompt_idx + args.prompt_start_index])['prompt']

        # set watermark strength
        if args.watermark == 'kgw':
            watermark.config.delta = 3.0
        elif args.watermark == 'aar':
            watermark.config.temperature = 0.7
        
        # generate watermarked text
        watermarked_text = watermark.generate_watermarked_text(prompt)
        watermarked_tokens = tokenizer.encode(watermarked_text, return_tensors='pt', 
                                            add_special_tokens=False).squeeze()
        
        # truncate prompt
        prompt_tokens = tokenizer.encode(prompt, return_tensors='pt', 
                                       add_special_tokens=False).squeeze()
        watermarked_tokens = watermarked_tokens[len(prompt_tokens):]
        
        # random length truncation
        random_length = torch.randint(args.min_length, args.max_length + 1, (1,)).item()
        watermarked_tokens_fragment = watermarked_tokens[:random_length]

        # paraphrase
        watermarked_text = tokenizer.decode(watermarked_tokens_fragment, skip_special_tokens=True)
        para_watermarked_text = attacker.edit(watermarked_text)
        
        # Skip this sample if attack failed
        if not para_watermarked_text:
            prompt_idx += world_size
            continue
            
        para_watermarked_tokens_fragment = tokenizer.encode(para_watermarked_text, 
                                                          return_tensors='pt', 
                                                          add_special_tokens=False).squeeze()

        # insert into wiki text
        if wiki_idx >= len(wiki_data):
            wiki_idx = rank  # Reset to initial offset if we run out of wiki data
            
        wiki_tokens = tokenizer.encode(wiki_data[wiki_idx + args.wiki_start_index]['text'], 
                                return_tensors='pt', 
                                add_special_tokens=False).squeeze()[:args.wiki_length]
        insert_position = torch.randint(0, len(wiki_tokens), (1,)).item()
        
        tokens = torch.cat((wiki_tokens[:insert_position], watermarked_tokens_fragment, wiki_tokens[insert_position:]))
        para_tokens = torch.cat((wiki_tokens[:insert_position], para_watermarked_tokens_fragment, wiki_tokens[insert_position:]))
        
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        para_text = tokenizer.decode(para_tokens, skip_special_tokens=True)
        
        data.append({
            'text': text, 
            'flag': 1, 
            'start_index': insert_position, 
            'end_index': insert_position+random_length, 
            'strength': watermark.config.delta if args.watermark == 'kgw' else watermark.config.temperature
        })
        
        para_data.append({
            'text': para_text, 
            'flag': 1, 
            'start_index': insert_position, 
            'end_index': insert_position+len(para_watermarked_tokens_fragment), 
            'strength': watermark.config.delta if args.watermark == 'kgw' else watermark.config.temperature
        })
        
        # Update counter atomically
        while True:
            try:
                with open(counter_file, 'r') as f:
                    current_count = int(f.read().strip())
                with open(counter_file, 'w') as f:
                    f.write(str(current_count + 1))
                break
            except (FileNotFoundError, ValueError):
                with open(counter_file, 'w') as f:
                    f.write('1')
                break
        
        prompt_idx += world_size
        wiki_idx += world_size
        pbar.update(1)
    
    # save data for this GPU
    output_file = args.output_file.replace('.json', f'_gpu{rank}.json')
    para_output_file = args.output_file.replace('.json', f'_{args.attack_type}_gpu{rank}.json')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for d in data:
            f.write(json.dumps(d)+'\n')
            
    with open(para_output_file, 'w') as f:
        for d in para_data:
            f.write(json.dumps(d)+'\n')

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
    parser.add_argument('--attack_type', type=str, default='paraphrase')
    parser.add_argument('--output_file', type=str, default='data/kgw/llama/data_delta_1.json')
    parser.add_argument('--model', type=str, default='llama')
    args = parser.parse_args()

    # Initialize counter file
    counter_file = os.path.join(os.path.dirname(args.output_file), 'sample_counter.txt')
    with open(counter_file, 'w') as f:
        f.write('0')

    world_size = torch.cuda.device_count()
    
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=world_size) as executor:
        for rank in range(world_size):
            future = executor.submit(
                generate_data_for_gpu,
                rank,
                args,
                world_size
            )
            futures.append(future)
        
        concurrent.futures.wait(futures)

    # Clean up counter file
    os.remove(counter_file)

    # Merge output files
    print("Merging output files...")
    for suffix in ['', f'_{args.attack_type}']:
        all_data = []
        for rank in range(world_size):
            gpu_output_file = args.output_file.replace('.json', f'{suffix}_gpu{rank}.json')
            if os.path.exists(gpu_output_file):
                with open(gpu_output_file, 'r') as f:
                    for line in f:
                        all_data.append(json.loads(line))
                os.remove(gpu_output_file)
        
        output_file = args.output_file.replace('.json', f'{suffix}.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for data in all_data:
                f.write(json.dumps(data) + '\n')
        
        print(f"Data merged into {output_file}")