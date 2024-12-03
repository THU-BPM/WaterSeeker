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
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM

def generate_data_for_gpu(rank, args, start_idx, end_idx, wiki_lengths):

    device = f"cuda:{rank}"
    
    if args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained('/workspace/intern_ckpt/panleyi/Llama-2-7b-hf')

        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained('/workspace/intern_ckpt/panleyi/Llama-2-7b-hf', device_map=device),
                                            tokenizer=tokenizer,
                                            vocab_size=32000,
                                            device=device,
                                            max_new_tokens=args.max_length,
                                            min_length=args.max_length + 30,
                                            no_repeat_ngram_size=4,
                                            do_sample=True)

    elif args.model == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained('/workspace/intern_ckpt/panleyi/Mistral-7B-v0.1')

        transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained('/workspace/intern_ckpt/panleyi/Mistral-7B-v0.1', device_map=device),
                                                tokenizer=tokenizer,
                                                vocab_size=32000,
                                                device=device,
                                                max_new_tokens=args.max_length,
                                                min_length=args.max_length + 30,
                                                no_repeat_ngram_size=4,
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
    
    data = [[] for _ in range(len(wiki_lengths))]
    
    pbar = tqdm(total=end_idx-start_idx, desc=f'GPU {rank}')

    # generate watermarked data
    for i in range(start_idx, end_idx):
        prompt = json.loads(prompt_data[i + args.prompt_start_index])['prompt']

        # random sample watermark strength
        if args.watermark == 'kgw':
            watermark.config.delta = random.choice([2.0, 1.5, 1.0])
        elif args.watermark == 'aar':
            watermark.config.temperature = random.choice([0.5, 0.4, 0.3])
        
        # generate watermarked text
        watermarked_text = watermark.generate_watermarked_text(prompt)
        watermarked_tokens = tokenizer.encode(watermarked_text, return_tensors='pt', 
                                            add_special_tokens=False).squeeze()
        
        # truncate prompt from watermarked tokens
        prompt_tokens = tokenizer.encode(prompt, return_tensors='pt', 
                                       add_special_tokens=False).squeeze()
        watermarked_tokens = watermarked_tokens[len(prompt_tokens):]
        
        # truncate watermarked tokens to a random length
        random_length = torch.randint(args.min_length, args.max_length + 1, (1,)).item()
        watermarked_tokens_fragment = watermarked_tokens[:random_length]

        # insert watermarked text into wiki text
        for idx in range(len(wiki_lengths)):
            wiki_length = wiki_lengths[idx]
            wiki_tokens = tokenizer.encode(wiki_data[i + args.wiki_start_index]['text'], 
                                    return_tensors='pt', add_special_tokens=False).squeeze()[:wiki_length]
            insert_position = torch.randint(0, len(wiki_tokens), (1,)).item()
            tokens = torch.cat((wiki_tokens[:insert_position], watermarked_tokens_fragment, wiki_tokens[insert_position:]))
            text = tokenizer.decode(tokens, skip_special_tokens=True)
        
            segment_info = [(insert_position, insert_position+random_length)]
            data[idx].append({
                'text': text, 
                'flag': 1, 
                'segments': segment_info,
                'strength': watermark.config.delta if args.watermark == 'kgw' else watermark.config.temperature
            })
        
        pbar.update(1)
    
    # save data for this GPU
    for idx in range(len(wiki_lengths)):
        wiki_length = wiki_lengths[idx]
        output_file = args.output_file.replace('.json', f'{wiki_length}_gpu{rank}.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for d in data[idx]:
                f.write(json.dumps(d)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for watermarking')
    parser.add_argument('--num_sample', type=int, default=300)
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

    wiki_lengths = [500, 2000, 5000, 10000]

    world_size = torch.cuda.device_count()
    samples_per_gpu = args.num_sample // world_size
    
    futures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=world_size) as executor:
        for rank in range(world_size):
            start_idx = rank * samples_per_gpu
            end_idx = start_idx + samples_per_gpu if rank != world_size-1 else args.num_sample
            
            future = executor.submit(
                generate_data_for_gpu,
                rank,
                args,
                start_idx,
                end_idx,
                wiki_lengths
            )
            futures.append(future)
        
        # Wait for all processes to complete and check for exceptions
        try:
            results = []
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())  # This will raise any exceptions that occurred
        except Exception as e:
            print(f"An error occurred during processing: {e}")
            raise

    # 确保所有GPU的数据都已生成完成后再合并
    print("Checking if all GPU files exist...")
    for length in wiki_lengths:
        for rank in range(world_size):
            gpu_output_file = args.output_file.replace('.json', f'{length}_gpu{rank}.json')
            if not os.path.exists(gpu_output_file):
                raise FileNotFoundError(f"Missing output file from GPU {rank} for length {length}")

    # 合并文件
    print("All GPU files found. Merging output files...")
    for length in wiki_lengths:
        all_data = []
        for rank in range(world_size):
            gpu_output_file = args.output_file.replace('.json', f'{length}_gpu{rank}.json')
            with open(gpu_output_file, 'r') as f:
                for line in f:
                    all_data.append(json.loads(line))
            # Remove temporary files
            os.remove(gpu_output_file)
        
        # Write final output file for each length
        output_file = args.output_file.replace('.json', f'_{length}.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            for data in all_data:
                f.write(json.dumps(data) + '\n')
        
        print(f"Data for length {length} merged into {output_file}")