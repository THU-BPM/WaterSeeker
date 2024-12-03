export PYTHONPATH="/workspace/panleyi/WaterSeeker:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python seeker/seeker_ablation.py --watermark kgw \
    --targeted_fpr 1e-6 \
    --input_file data/main/kgw_llama_10000.json \
    --output_file baseline_result/seeker_ablation/kgw_llama.log \
    --window_size 50 \
    --threshold_1 0.5 \
    --threshold_2 0.0 \
    --min_length 100 \
    --model llama \
    --key 33554393

python evaluate_ablation.py --watermark kgw \
    --input_file baseline_result/seeker_ablation/kgw_llama.log \
    --iou_threshold 0.0 \
    --detection_method seeker

CUDA_VISIBLE_DEVICES=0 python seeker/seeker_ablation.py --watermark aar \
    --targeted_fpr 1e-6 \
    --input_file data/main/aar_llama_10000.json \
    --output_file baseline_result/seeker_ablation/aar_llama.log \
    --window_size 50 \
    --threshold_1 0.5 \
    --threshold_2 0.0 \
    --min_length 100 \
    --model llama \
    --seed 42

python evaluate_ablation.py --watermark aar \
    --input_file baseline_result/seeker_ablation/aar_llama.log \
    --iou_threshold 0.0 \
    --detection_method seeker