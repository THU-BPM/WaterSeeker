# generate data
CUDA_VISIBLE_DEVICES=0,2 python -u generate_data.py --num_sample 300 --wiki_start_index 0 --prompt_start_index 0 --watermark kgw --key 33554393 --model llama --output_file data/kgw/llama/data_1.json

# full-text
CUDA_VISIBLE_DEVICES=2,3 python -u baselines/full_text.py --watermark kgw --input_file data/kgw/llama/data.json --output_file baseline_result/llama/kgw_full_text.log --model llama --key 33554393

# winmax
CUDA_VISIBLE_DEVICES=2,3 python -u baselines/winmax.py --watermark kgw --input_file data/kgw/llama/data.json --output_file baseline_result/llama/kgw_winmax.log --min_window_length 100 --max_window_length 400 --model llama --key 33554393

# seeker
CUDA_VISIBLE_DEVICES=2,3 python -u seeker/seeker.py --watermark kgw --targeted_fpr 1e-6 --input_file data/kgw/llama/data.json --output_file baseline_result/llama/kgw_seeker.log --window_size 50 --min_length 100 --model llama --key 33554393

# flsw
CUDA_VISIBLE_DEVICES=2,3 python -u baselines/fix_window.py --watermark kgw --input_file data/kgw/llama/data.json --window_size 100 --output_file baseline_result/llama/kgw_fix_window_100.log --model llama --key 33554393

# evaluate
python evaluate.py --watermark kgw --input_file baseline_result/llama/kgw_seeker.log --watermark kgw --iou_threshold 0.5 --detection_method seeker