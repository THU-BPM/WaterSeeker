# nohup bash 2_winmax_aar.sh > 2_winmax_aar.log 2>&1 &
# winmax
## aar-llama
# python baselines/winmax_parallel.py --watermark aar \
#     --input_file data/main/aar_llama_10000.json \
#     --min_window_length 100 \
#     --max_window_length 400 \
#     --window_interval 1 \
#     --seed 42 \
#     --output_file baseline_result/winmax_main/aar_llama_1.log \
#     --model llama \
#     --num_workers 6

python evaluate.py --watermark aar \
    --input_file baseline_result/winmax_main/aar_llama_1.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark aar \
    --input_file data/main/aar_llama_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 20 \
    --seed 42 \
    --output_file baseline_result/winmax_main/aar_llama_20.log \
    --model llama \
    --num_workers 6

python evaluate.py --watermark aar \
    --input_file baseline_result/winmax_main/aar_llama_20.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark aar \
    --input_file data/main/aar_llama_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 50 \
    --seed 42 \
    --output_file baseline_result/winmax_main/aar_llama_50.log \
    --model llama \
    --num_workers 6

python evaluate.py --watermark aar \
    --input_file baseline_result/winmax_main/aar_llama_50.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark aar \
    --input_file data/main/aar_llama_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 100 \
    --seed 42 \
    --output_file baseline_result/winmax_main/aar_llama_100.log \
    --model llama \
    --num_workers 6

python baselines/winmax_parallel.py --watermark aar \
    --input_file data/main/aar_llama_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 200 \
    --seed 42 \
    --output_file baseline_result/winmax_main/aar_llama_200.log \
    --model llama \
    --num_workers 6

python evaluate.py --watermark aar \
    --input_file baseline_result/winmax_main/aar_llama_200.log \
    --iou_threshold 0.0 \
    --detection_method winmax

## aar-mistral
python baselines/winmax_parallel.py --watermark aar \
    --input_file data/main/aar_mistral_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 1 \
    --seed 42 \
    --output_file baseline_result/winmax_main/aar_mistral_1.log \
    --model mistral \
    --num_workers 6

python evaluate.py --watermark aar \
    --input_file baseline_result/winmax_main/aar_mistral_1.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark aar \
    --input_file data/main/aar_mistral_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 20 \
    --seed 42 \
    --output_file baseline_result/winmax_main/aar_mistral_20.log \
    --model mistral \
    --num_workers 6

python baselines/winmax_parallel.py --watermark aar \
    --input_file data/main/aar_mistral_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 50 \
    --seed 42 \
    --output_file baseline_result/winmax_main/aar_mistral_50.log \
    --model mistral \
    --num_workers 6

python evaluate.py --watermark aar \
    --input_file baseline_result/winmax_main/aar_mistral_50.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark aar \
    --input_file data/main/aar_mistral_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 100 \
    --seed 42 \
    --output_file baseline_result/winmax_main/aar_mistral_100.log \
    --model mistral \
    --num_workers 6

python evaluate.py --watermark aar \
    --input_file baseline_result/winmax_main/aar_mistral_100.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark aar \
    --input_file data/main/aar_mistral_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 200 \
    --seed 42 \
    --output_file baseline_result/winmax_main/aar_mistral_200.log \
    --model mistral \
    --num_workers 6

python evaluate.py --watermark aar \
    --input_file baseline_result/winmax_main/aar_mistral_200.log \
    --iou_threshold 0.0 \
    --detection_method winmax

## waterseeker
# CUDA_VISIBLE_DEVICES=0 python seeker/seeker.py --watermark aar \
#     --targeted_fpr 1e-6 \
#     --input_file data/main/aar_llama_10000.json \
#     --output_file baseline_result/seeker_main/aar_llama.log \
#     --window_size 50 \
#     --threshold_1 0.5 \
#     --threshold_2 0.0 \
#     --min_length 100 \
#     --model llama \
#     --seed 42

# python evaluate.py --watermark aar \
#     --input_file baseline_result/seeker_main/aar_llama.log \
#     --iou_threshold 0.0 \
#     --detection_method seeker

# CUDA_VISIBLE_DEVICES=1 python seeker/seeker.py --watermark aar \
#     --targeted_fpr 1e-6 \
#     --input_file data/main/aar_mistral_10000.json \
#     --output_file baseline_result/seeker_main/aar_mistral.log \
#     --window_size 50 \
#     --threshold_1 0.5 \
#     --threshold_2 0.0 \
#     --min_length 100 \
#     --model mistral \
#     --seed 42

# python evaluate.py --watermark aar \
#     --input_file baseline_result/seeker_main/aar_mistral.log \
#     --iou_threshold 0.0 \
#     --detection_method seeker
