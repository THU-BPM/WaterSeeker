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

def generate_data_for_gpu(rank, args, samples_per_gpu, start_idx, end_idx):
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
    
    data = []
    prompt_index = args.prompt_start_index + start_idx * args.num_segments
    wiki_index = start_idx

    pbar = tqdm(total=end_idx-start_idx, desc=f'GPU {rank}')

    # generate watermarked data
    while wiki_index < end_idx:
        # get wiki text tokens
        tokens = tokenizer.encode(wiki_data[wiki_index + args.wiki_start_index]['text'], 
                                return_tensors='pt', add_special_tokens=False).squeeze()[:args.wiki_length]
        
        segments_info = []

        # random sample watermark strength
        if args.watermark == 'kgw':
            watermark.config.delta = random.choice([2.0, 1.5, 1.0])
        elif args.watermark == 'aar':
            watermark.config.temperature = random.choice([0.5, 0.4, 0.3]) # change

        # insert multiple watermark segments
        for _ in range(args.num_segments):
            if prompt_index >= len(prompt_data):
                break
                
            prompt = json.loads(prompt_data[prompt_index])['prompt']
            
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

            # find valid insert position
            while True:
                insert_position = torch.randint(0, len(tokens), (1,)).item()
                
                # check if position is valid (respects minimum gap)
                is_valid = True
                for start, end in segments_info:
                    if (abs(insert_position - start) < args.min_segment_gap or 
                        abs(insert_position - end) < args.min_segment_gap):
                        is_valid = False
                        break
                
                if is_valid:
                    break

            # insert watermarked segment
            tokens = torch.cat((tokens[:insert_position], watermarked_tokens_fragment, tokens[insert_position:]))
            
            # update previously inserted segments' positions
            for i in range(len(segments_info)):
                start, end = segments_info[i]
                if start >= insert_position:
                    segments_info[i] = (start + len(watermarked_tokens_fragment), 
                                      end + len(watermarked_tokens_fragment))
            
            # add new segment info
            segments_info.append((insert_position, insert_position + len(watermarked_tokens_fragment)))
            prompt_index += 1

        if len(segments_info) == args.num_segments:
            # decode final text
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            
            # sort segments by start position for output
            segments_info.sort(key=lambda x: x[0])
            
            data.append({
                'text': text, 
                'flag': 1, 
                'segments': segments_info,
                'strength': watermark.config.delta if args.watermark == 'kgw' else watermark.config.temperature
            })
            wiki_index += 1
            pbar.update(1)
    
    # save data for this GPU
    output_file = args.output_file.replace('.json', f'_gpu{rank}.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for d in data:
            f.write(json.dumps(d)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for watermarking')
    parser.add_argument('--num_sample', type=int, default=300)
    parser.add_argument('--num_segments', type=int, default=3,
                        help='Number of watermark segments to insert')
    parser.add_argument('--min_segment_gap', type=int, default=200,
                        help='Minimum gap between watermark segments')
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
                samples_per_gpu,
                start_idx,
                end_idx
            )
            futures.append(future)
        
        # Wait for all processes to complete
        concurrent.futures.wait(futures)

    # Merge output files from all GPUs
    print("Merging output files...")
    all_data = []
    for rank in range(world_size):
        gpu_output_file = args.output_file.replace('.json', f'_gpu{rank}.json')
        if os.path.exists(gpu_output_file):
            with open(gpu_output_file, 'r') as f:
                for line in f:
                    all_data.append(json.loads(line))
            # Remove temporary files
            os.remove(gpu_output_file)
    
    # Write final output file
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for data in all_data:
            f.write(json.dumps(data) + '\n')
    
    print(f"All data merged into {args.output_file}")