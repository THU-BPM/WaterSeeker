# ==================================================================
# winmax.py
# Description: WinMax baseline for watermarked segments detection with parallel processing.
# ==================================================================

import os
import json
import torch
import argparse
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from watermark.kgw import KGW
from watermark.aar import AAR
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer

def process_batch(batch_data, worker_id, args):
    """Process a batch of texts with one model instance."""
    # Initialize model and watermark for this worker
    print(f"\nWorker {worker_id} initializing model...")
    watermark = init_model_and_watermark(args)
    
    results = []
    # 为每个worker创建独立的进度条
    pbar = tqdm(total=len(batch_data), 
                desc=f"Worker {worker_id}",
                position=worker_id)
    
    for text_data in batch_data:
        text = text_data['text']
        flag = text_data['flag']
        segments = text_data['segments']

        detect_result = watermark.detect_watermark_win_max(
            text=text, 
            min_L=args.min_window_length, 
            max_L=args.max_window_length,
            window_interval=args.window_interval
        )
        
        results.append({
            'predicted': detect_result['is_watermarked'],
            'indices': detect_result['indices'],
            'gold': flag,
            'gold_indices': segments
        })
        
        pbar.update(1)
    
    pbar.close()
    return results, len(batch_data)

def init_model_and_watermark(args):
    """Initialize model, tokenizer and watermark."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained('/workspace/intern_ckpt/panleyi/Llama-2-7b-hf')
        transformers_config = TransformersConfig(
            model=AutoModelForCausalLM.from_pretrained('/workspace/intern_ckpt/panleyi/Llama-2-7b-hf', device_map='auto'),
            tokenizer=tokenizer,
            vocab_size=32000,
            device=device
        )
    elif args.model == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained('/workspace/intern_ckpt/panleyi/Mistral-7B-v0.1')
        transformers_config = TransformersConfig(
            model=AutoModelForCausalLM.from_pretrained('/workspace/intern_ckpt/panleyi/Mistral-7B-v0.1', device_map='auto'),
            tokenizer=tokenizer,
            vocab_size=32000,
            device=device
        )

    if args.watermark == 'kgw':
        watermark = KGW('config/KGW.json', transformers_config=transformers_config)
        watermark.config.hash_key = args.key
    elif args.watermark == 'aar':
        watermark = AAR('config/AAR.json', transformers_config=transformers_config)
        watermark.config.seed = args.seed

    return watermark

def split_data(data, num_workers):
    """Split data into approximately equal chunks for each worker."""
    chunk_size = len(data) // num_workers
    remainder = len(data) % num_workers
    
    chunks = []
    start = 0
    for i in range(num_workers):
        # Add one extra item to some chunks if there's remainder
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start:end])
        start = end
        
    return chunks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate data for watermarking')
    parser.add_argument('--watermark', type=str, default='kgw')
    parser.add_argument('--input_file', type=str, default='data/non_watermarked_data.json')
    parser.add_argument('--min_window_length', type=int, default=100)
    parser.add_argument('--max_window_length', type=int, default=400)
    parser.add_argument('--window_interval', type=int, default=1)
    parser.add_argument('--key', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_file', type=str, default='baseline_result/kgw_winmax.log')
    parser.add_argument('--model', type=str, default='llama')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    args = parser.parse_args()

    # Read input data
    with open(args.input_file) as f:
        data = [json.loads(line) for line in f]

    # Split data into chunks for each worker
    data_chunks = split_data(data, args.num_workers)
    print(f"Split {len(data)} samples into {args.num_workers} chunks:")
    for i, chunk in enumerate(data_chunks):
        print(f"Worker {i}: {len(chunk)} samples")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Process data in parallel
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit batch processing tasks
        future_to_worker = {
            executor.submit(process_batch, chunk, worker_id, args): worker_id 
            for worker_id, chunk in enumerate(data_chunks)
        }

        # Create overall progress bar
        overall_pbar = tqdm(total=len(data), 
                          desc="Overall Progress",
                          position=args.num_workers)

        # Collect results
        all_results = []
        completed = 0
        for future in as_completed(future_to_worker):
            worker_id = future_to_worker[future]
            try:
                batch_results, batch_size = future.result()
                all_results.extend(batch_results)
                completed += batch_size
                overall_pbar.update(batch_size)
            except Exception as e:
                print(f"Error in worker {worker_id}: {e}")

        overall_pbar.close()

        # Move cursor to the bottom of all progress bars
        print("\n" * (args.num_workers + 2))

        # Write all results to file
        print(f"Writing {len(all_results)} results to {args.output_file}")
        with open(args.output_file, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')