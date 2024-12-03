# export PYTHONPATH="/workspace/panleyi/WaterSeeker:$PYTHONPATH"

# nohup bash 8_robustness.sh > 8_robustness.log 2>&1 &

# normal
python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/robustness/data_delta_3.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 50 \
    --key 33554393 \
    --output_file baseline_result/robustness/normal/winmax_50.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/robustness/normal/winmax_50.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/robustness/data_delta_3.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 100 \
    --key 33554393 \
    --output_file baseline_result/robustness/normal/winmax_100.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/robustness/normal/winmax_100.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/robustness/data_delta_3.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 200 \
    --key 33554393 \
    --output_file baseline_result/robustness/normal/winmax_200.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/robustness/normal/winmax_200.log \
    --iou_threshold 0.0 \
    --detection_method winmax

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/normal/winmax_1.log \
#     --iou_threshold 0.0 \
#     --detection_method winmax

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/normal/seeker.log \
#     --iou_threshold 0.0 \
#     --detection_method seeker

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/normal/flsw_100.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/normal/flsw_200.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/normal/flsw_300.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/normal/flsw_400.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# substitution
python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/robustness/data_delta_3_substitution.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 50 \
    --key 33554393 \
    --output_file baseline_result/robustness/substitution/winmax_50.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/robustness/substitution/winmax_50.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/robustness/data_delta_3_substitution.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 100 \
    --key 33554393 \
    --output_file baseline_result/robustness/substitution/winmax_100.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/robustness/substitution/winmax_100.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/robustness/data_delta_3_substitution.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 200 \
    --key 33554393 \
    --output_file baseline_result/robustness/substitution/winmax_200.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/robustness/substitution/winmax_200.log \
    --iou_threshold 0.0 \
    --detection_method winmax

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/substitution/winmax_1.log \
#     --iou_threshold 0.0 \
#     --detection_method winmax

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/substitution/seeker.log \
#     --iou_threshold 0.0 \
#     --detection_method seeker

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/substitution/flsw_100.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/substitution/flsw_200.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/substitution/flsw_300.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/substitution/flsw_400.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# deletion
python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/robustness/data_delta_3_deletion.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 50 \
    --key 33554393 \
    --output_file baseline_result/robustness/deletion/winmax_50.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/robustness/deletion/winmax_50.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/robustness/data_delta_3_deletion.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 100 \
    --key 33554393 \
    --output_file baseline_result/robustness/deletion/winmax_100.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/robustness/deletion/winmax_100.log \
    --iou_threshold 0.0 \
    --detection_method winmax

python baselines/winmax_parallel.py --watermark kgw \
    --input_file data/robustness/data_delta_3_deletion.json \
    --min_window_length 100 \
    --max_window_length 400 \
    --window_interval 200 \
    --key 33554393 \
    --output_file baseline_result/robustness/deletion/winmax_200.log \
    --model llama \
    --num_workers 8

python evaluate.py --watermark kgw \
    --input_file baseline_result/robustness/deletion/winmax_200.log \
    --iou_threshold 0.0 \
    --detection_method winmax

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/deletion/winmax_1.log \
#     --iou_threshold 0.0 \
#     --detection_method winmax

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/deletion/seeker.log \
#     --iou_threshold 0.0 \
#     --detection_method seeker

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/deletion/flsw_100.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/deletion/flsw_200.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/deletion/flsw_300.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/robustness/deletion/flsw_400.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw
