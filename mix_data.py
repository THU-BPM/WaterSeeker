import json
import argparse
from tqdm import tqdm
from transformers import LlamaTokenizer, AutoTokenizer

def main(args):
    if args.model == 'llama':
        tokenizer = LlamaTokenizer.from_pretrained('/workspace/intern_ckpt/panleyi/Llama-2-7b-hf')
    elif args.model == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained('/workspace/intern_ckpt/panleyi/Mistral-7B-v0.1')

    # 读取non_wat_file
    processed_non_wat_data = []
    with open(args.non_wat_file, 'r') as f:
        non_wat_data = f.readlines()
    for line in tqdm(non_wat_data):
        data = json.loads(line)
        text = data['text']
        if args.length != 10000:
            encoded_text = tokenizer.encode(text, return_tensors='pt', add_special_tokens=False)[:args.length].squeeze()
            text = tokenizer.decode(encoded_text, skip_special_tokens=True)
        processed_non_wat_data.append({'text': text, 'flag': 0, 'segments': []})
    
    # 将processed_non_wat_data追加写入wat_file
    with open(args.wat_file, 'a') as f:
        for data in processed_non_wat_data:
            f.write(json.dumps(data) + '\n')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wat_file', type=str, required=True)
    parser.add_argument('--non_wat_file', type=str, required=True)
    parser.add_argument('--length', type=int, default=10000)
    parser.add_argument('--model', type=str, default='llama')
    args = parser.parse_args()

    main(args)