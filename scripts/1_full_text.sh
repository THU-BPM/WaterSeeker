export PYTHONPATH="/workspace/panleyi/WaterSeeker:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=0 python baselines/full_text.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --output_file baseline_result/full_text/kgw_llama.log \
    --model llama \
    --key 33554393

CUDA_VISIBLE_DEVICES=1 python baselines/full_text.py --watermark kgw \
    --input_file data/main/kgw_mistral_10000.json \
    --output_file baseline_result/full_text/kgw_mistral.log \
    --model mistral \
    --key 4294967291

python evaluate.py --watermark kgw \
    --input_file baseline_result/full_text/kgw_llama.log \
    --watermark kgw \
    --iou_threshold 0.0 \
    --detection_method full

python evaluate.py --watermark kgw \
    --input_file baseline_result/full_text/kgw_mistral.log \
    --watermark kgw \
    --iou_threshold 0.0 \
    --detection_method full

CUDA_VISIBLE_DEVICES=0 python baselines/full_text.py --watermark aar \
    --input_file data/main/aar_llama_10000.json \
    --output_file baseline_result/full_text/aar_llama.log \
    --model llama \
    --seed 42

python evaluate.py --watermark aar \
    --input_file baseline_result/full_text/aar_llama.log \
    --watermark aar \
    --iou_threshold 0.0 \
    --detection_method full

CUDA_VISIBLE_DEVICES=1 python baselines/full_text.py --watermark aar \
    --input_file data/main/aar_mistral_10000.json \
    --output_file baseline_result/full_text/aar_mistral.log \
    --model mistral \
    --seed 42

python evaluate.py --watermark aar \
    --input_file baseline_result/full_text/aar_mistral.log \
    --watermark aar \
    --iou_threshold 0.0 \
    --detection_method full