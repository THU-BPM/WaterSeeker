export PYTHONPATH="/workspace/panleyi/WaterSeeker:$PYTHONPATH"

# winmax
## kgw-llama
python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 1 \
    --key 33554393 \
    --output_file baseline_result/winmax_main/kgw_llama_1.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_llama_1.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 20 \
    --key 33554393 \
    --output_file baseline_result/winmax_main/kgw_llama_20.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_llama_20.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 50 \
    --key 33554393 \
    --output_file baseline_result/winmax_main/kgw_llama_50.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_llama_50.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 100 \
    --key 33554393 \
    --output_file baseline_result/winmax_main/kgw_llama_100.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_llama_100.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --min_window_length 100 \
    --max_window_length 500 \
    --window_interval 150 \
    --key 33554393 \
    --output_file baseline_result/winmax_main/kgw_llama_150.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_llama_150.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 200 \
    --key 33554393 \
    --output_file baseline_result/winmax_main/kgw_llama_200.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_llama_200.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --min_window_length 100 \
    --max_window_length 500 \
    --window_interval 250 \
    --key 33554393 \
    --output_file baseline_result/winmax_main/kgw_llama_250.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_llama_250.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --min_window_length 100 \
    --max_window_length 600 \
    --window_interval 300 \
    --key 33554393 \
    --output_file baseline_result/winmax_main/kgw_llama_300.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_llama_300.log \
    --iou_threshold 0.0 \
    --detection_method winmax

## kgw-mistral
python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_mistral_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 1 \
    --key 4294967291 \
    --output_file baseline_result/winmax_main/kgw_mistral_1.log \
    --model mistral \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_mistral_1.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_mistral_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 20 \
    --key 4294967291 \
    --output_file baseline_result/winmax_main/kgw_mistral_20.log \
    --model mistral \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_mistral_20.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_mistral_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 50 \
    --key 4294967291 \
    --output_file baseline_result/winmax_main/kgw_mistral_50.log \
    --model mistral \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_mistral_50.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_mistral_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 100 \
    --key 4294967291 \
    --output_file baseline_result/winmax_main/kgw_mistral_100.log \
    --model mistral \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_mistral_100.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_mistral_10000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 200 \
    --key 4294967291 \
    --output_file baseline_result/winmax_main/kgw_mistral_200.log \
    --model mistral \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_main/kgw_mistral_200.log \
    --iou_threshold 0.0 \
    --detection_method winmax

## aar-llama


# waterseeker
CUDA_VISIBLE_DEVICES=0 python seeker/seeker.py --watermark kgw \
    --targeted_fpr 1e-6 \
    --input_file data/main/kgw_llama_10000.json \
    --output_file baseline_result/seeker_main/kgw_llama.log \
    --window_size 50 \
    --threshold_1 0.5 \
    --threshold_2 0.0 \
    --min_length 100 \
    --model llama \
    --key 33554393

python evaluate.py --watermark kgw \
    --input_file baseline_result/seeker_main/kgw_llama.log \
    --iou_threshold 0.0 \
    --detection_method seeker

CUDA_VISIBLE_DEVICES=1 python seeker/seeker.py --watermark kgw \
    --targeted_fpr 1e-6 \
    --input_file data/main/kgw_mistral_10000.json \
    --output_file baseline_result/seeker_main/kgw_mistral.log \
    --window_size 50 \
    --threshold_1 0.5 \
    --threshold_2 0.0 \
    --min_length 100 \
    --model mistral \
    --key 4294967291

python evaluate.py --watermark kgw \
    --input_file baseline_result/seeker_main/kgw_mistral.log \
    --iou_threshold 0.0 \
    --detection_method seeker


