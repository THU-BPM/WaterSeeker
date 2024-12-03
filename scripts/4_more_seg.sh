# export PYTHONPATH="/workspace/panleyi/WaterSeeker:$PYTHONPATH"
# nohup bash 4_more_seg.sh > 4_more_seg.log 2>&1 &
# python -u generate_data_more_seg.py --num_sample 300 \
#     --num_segment 3 \
#     --min_segment_gap 200 \
#     --wiki_start_index 0 \
#     --prompt_start_index 0 \
#     --watermark aar \
#     --seed 42 \
#     --model llama \
#     --output_file data/3_segments/aar_llama.json

# # kgw-llama
# python evaluate.py --watermark kgw \
#     --input_file baseline_result/3_segments/kgw_llama/flsw_100.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/3_segments/kgw_llama/flsw_200.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/3_segments/kgw_llama/flsw_300.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/3_segments/kgw_llama/flsw_400.log \
#     --iou_threshold 0.0 \
#     --detection_method flsw

# python evaluate.py --watermark kgw \
#     --input_file baseline_result/3_segments/kgw_llama/seeker.log \
#     --iou_threshold 0.0 \
#     --detection_method seeker

# aar-llama
CUDA_VISIBLE_DEVICES=0 python baselines/fix_window.py --watermark aar \
    --input_file data/3_segments/aar_llama.json \
    --targeted_fpr 1e-6 \
    --window_size 100 \
    --output_file baseline_result/3_segments/aar_llama/flsw_100.log \
    --model llama \
    --seed 42

python evaluate.py --watermark aar \
    --input_file baseline_result/3_segments/aar_llama/flsw_100.log \
    --iou_threshold 0.0 \
    --detection_method flsw

CUDA_VISIBLE_DEVICES=0 python baselines/fix_window.py --watermark aar \
    --input_file data/3_segments/aar_llama.json \
    --targeted_fpr 1e-6 \
    --window_size 200 \
    --output_file baseline_result/3_segments/aar_llama/flsw_200.log \
    --model llama \
    --seed 42

python evaluate.py --watermark aar \
    --input_file baseline_result/3_segments/aar_llama/flsw_200.log \
    --iou_threshold 0.0 \
    --detection_method flsw

CUDA_VISIBLE_DEVICES=1 python baselines/fix_window.py --watermark aar \
    --input_file data/3_segments/aar_llama.json \
    --targeted_fpr 1e-6 \
    --window_size 300 \
    --output_file baseline_result/3_segments/aar_llama/flsw_300.log \
    --model llama \
    --seed 42

python evaluate.py --watermark aar \
    --input_file baseline_result/3_segments/aar_llama/flsw_300.log \
    --iou_threshold 0.0 \
    --detection_method flsw

CUDA_VISIBLE_DEVICES=2 python baselines/fix_window.py --watermark aar \
    --input_file data/3_segments/aar_llama.json \
    --targeted_fpr 1e-6 \
    --window_size 400 \
    --output_file baseline_result/3_segments/aar_llama/flsw_400.log \
    --model llama \
    --seed 42

python evaluate.py --watermark aar \
    --input_file baseline_result/3_segments/aar_llama/flsw_400.log \
    --iou_threshold 0.0 \
    --detection_method flsw

CUDA_VISIBLE_DEVICES=0 python seeker/seeker.py --watermark aar \
    --targeted_fpr 1e-6 \
    --input_file data/3_segments/aar_llama.json \
    --output_file baseline_result/3_segments/aar_llama/seeker.log \
    --window_size 50 \
    --threshold_1 0.5 \
    --threshold_2 0.0 \
    --min_length 100 \
    --model llama \
    --seed 42

python evaluate.py --watermark aar \
    --input_file baseline_result/3_segments/aar_llama/seeker.log \
    --iou_threshold 0.0 \
    --detection_method seeker
