# export PYTHONPATH="/workspace/panleyi/WaterSeeker:$PYTHONPATH"
# `nohup bash 3_N.sh > 3_N.log 2>&1 &`
# N=5000
python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_5000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 1 \
    --key 33554393 \
    --output_file baseline_result/winmax_N/N_5000_int_1.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_N/N_5000_int_1.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_5000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 50 \
    --key 33554393 \
    --output_file baseline_result/winmax_N/N_5000_int_50.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_N/N_5000_int_50.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_5000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 100 \
    --key 33554393 \
    --output_file baseline_result/winmax_N/N_5000_int_100.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_N/N_5000_int_100.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_5000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 200 \
    --key 33554393 \
    --output_file baseline_result/winmax_N/N_5000_int_200.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_N/N_5000_int_200.log \
    --iou_threshold 0.0 \
    --detection_method winmax

CUDA_VISIBLE_DEVICES=0 python seeker/seeker.py --watermark kgw \
    --targeted_fpr 1e-6 \
    --input_file data/main/kgw_llama_5000.json \
    --output_file baseline_result/seeker_N/N_5000.log \
    --window_size 50 \
    --threshold_1 0.5 \
    --threshold_2 0.0 \
    --min_length 100 \
    --model llama \
    --key 33554393

CUDA_VISIBLE_DEVICES=0 python seeker/seeker.py --watermark kgw \
    --targeted_fpr 1e-6 \
    --input_file data/main/kgw_llama_2000.json \
    --output_file test/test.log \
    --window_size 50 \
    --threshold_1 0.5 \
    --threshold_2 0.0 \
    --min_length 100 \
    --model llama \
    --key 33554393

python evaluate.py --watermark kgw \
    --input_file baseline_result/seeker_N/N_5000.log \
    --iou_threshold 0.0 \
    --detection_method seeker

# N=2000
python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_2000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 1 \
    --key 33554393 \
    --output_file baseline_result/winmax_N/N_2000_int_1.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_N/N_2000_int_1.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_2000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 50 \
    --key 33554393 \
    --output_file baseline_result/winmax_N/N_2000_int_50.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_N/N_2000_int_50.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_2000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 100 \
    --key 33554393 \
    --output_file baseline_result/winmax_N/N_2000_int_100.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_N/N_2000_int_100.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_2000.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 200 \
    --key 33554393 \
    --output_file baseline_result/winmax_N/N_2000_int_200.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_N/N_2000_int_200.log \
    --iou_threshold 0.0 \
    --detection_method winmax

CUDA_VISIBLE_DEVICES=0 python seeker/seeker.py --watermark kgw \
    --targeted_fpr 1e-6 \
    --input_file data/main/kgw_llama_2000.json \
    --output_file baseline_result/seeker_N/N_2000.log \
    --window_size 50 \
    --threshold_1 0.5 \
    --threshold_2 0.0 \
    --min_length 100 \
    --model llama \
    --key 33554393

python evaluate.py --watermark kgw \
    --input_file baseline_result/seeker_N/N_2000.log \
    --iou_threshold 0.0 \
    --detection_method seeker

# N=500
python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_500.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 1 \
    --key 33554393 \
    --output_file baseline_result/winmax_N/N_500_int_1.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_N/N_500_int_1.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_500.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 50 \
    --key 33554393 \
    --output_file baseline_result/winmax_N/N_500_int_50.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_N/N_500_int_50.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_500.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 100 \
    --key 33554393 \
    --output_file baseline_result/winmax_N/N_500_int_100.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_N/N_500_int_100.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/main/kgw_llama_500.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 200 \
    --key 33554393 \
    --output_file baseline_result/winmax_N/N_500_int_200.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/winmax_N/N_500_int_200.log \
    --iou_threshold 0.0 \
    --detection_method winmax

CUDA_VISIBLE_DEVICES=0 python seeker/seeker.py --watermark kgw \
    --targeted_fpr 1e-6 \
    --input_file data/main/kgw_llama_500.json \
    --output_file baseline_result/seeker_N/N_500.log \
    --window_size 50 \
    --threshold_1 0.5 \
    --threshold_2 0.0 \
    --min_length 100 \
    --model llama \
    --key 33554393

python evaluate.py --watermark kgw \
    --input_file baseline_result/seeker_N/N_500.log \
    --iou_threshold 0.0 \
    --detection_method seeker