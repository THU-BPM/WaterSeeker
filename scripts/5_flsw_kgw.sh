export PYTHONPATH="/workspace/panleyi/WaterSeeker:$PYTHONPATH"

# kgw-llama
CUDA_VISIBLE_DEVICES=0 python baselines/fix_window.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --targeted_fpr 1e-6 \
    --window_size 100 \
    --output_file baseline_result/flsw_main/kgw_llama_100.log \
    --model llama \
    --key 33554393

python evaluate.py --watermark kgw \
    --input_file baseline_result/flsw_main/kgw_llama_100.log \
    --iou_threshold 0.0 \
    --detection_method flsw

CUDA_VISIBLE_DEVICES=0 python baselines/fix_window.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --targeted_fpr 1e-6 \
    --window_size 200 \
    --output_file baseline_result/flsw_main/kgw_llama_200.log \
    --model llama \
    --key 33554393

python evaluate.py --watermark kgw \
    --input_file baseline_result/flsw_main/kgw_llama_200.log \
    --iou_threshold 0.0 \
    --detection_method flsw

CUDA_VISIBLE_DEVICES=1 python baselines/fix_window.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --targeted_fpr 1e-6 \
    --window_size 300 \
    --output_file baseline_result/flsw_main/kgw_llama_300.log \
    --model llama \
    --key 33554393

python evaluate.py --watermark kgw \
    --input_file baseline_result/flsw_main/kgw_llama_300.log \
    --iou_threshold 0.0 \
    --detection_method flsw

CUDA_VISIBLE_DEVICES=2 python baselines/fix_window.py --watermark kgw \
    --input_file data/main/kgw_llama_10000.json \
    --targeted_fpr 1e-6 \
    --window_size 400 \
    --output_file baseline_result/flsw_main/kgw_llama_400.log \
    --model llama \
    --key 33554393

python evaluate.py --watermark kgw \
    --input_file baseline_result/flsw_main/kgw_llama_400.log \
    --iou_threshold 0.0 \
    --detection_method flsw

# kgw-mistral
CUDA_VISIBLE_DEVICES=3 python baselines/fix_window.py --watermark kgw \
    --input_file data/main/kgw_mistral_10000.json \
    --targeted_fpr 1e-6 \
    --window_size 100 \
    --output_file baseline_result/flsw_main/kgw_mistral_100.log \
    --model mistral \
    --key 4294967291

python evaluate.py --watermark kgw \
    --input_file baseline_result/flsw_main/kgw_mistral_100.log \
    --iou_threshold 0.0 \
    --detection_method flsw

CUDA_VISIBLE_DEVICES=4 python baselines/fix_window.py --watermark kgw \
    --input_file data/main/kgw_mistral_10000.json \
    --targeted_fpr 1e-6 \
    --window_size 200 \
    --output_file baseline_result/flsw_main/kgw_mistral_200.log \
    --model mistral \
    --key 4294967291

python evaluate.py --watermark kgw \
    --input_file baseline_result/flsw_main/kgw_mistral_200.log \
    --iou_threshold 0.0 \
    --detection_method flsw

CUDA_VISIBLE_DEVICES=5 python baselines/fix_window.py --watermark kgw \
    --input_file data/main/kgw_mistral_10000.json \
    --targeted_fpr 1e-6 \
    --window_size 300 \
    --output_file baseline_result/flsw_main/kgw_mistral_300.log \
    --model mistral \
    --key 4294967291

python evaluate.py --watermark kgw \
    --input_file baseline_result/flsw_main/kgw_mistral_300.log \
    --iou_threshold 0.0 \
    --detection_method flsw

CUDA_VISIBLE_DEVICES=6 python baselines/fix_window.py --watermark kgw \
    --input_file data/main/kgw_mistral_10000.json \
    --targeted_fpr 1e-6 \
    --window_size 400 \
    --output_file baseline_result/flsw_main/kgw_mistral_400.log \
    --model mistral \
    --key 4294967291

python evaluate.py --watermark kgw \
    --input_file baseline_result/flsw_main/kgw_mistral_400.log \
    --iou_threshold 0.0 \
    --detection_method flsw


