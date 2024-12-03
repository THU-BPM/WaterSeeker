# strong but short
python evaluate_flsw.py --watermark kgw \
    --input_file baseline_result/flsw_main/kgw_llama_300.log \
    --data_file data/main/kgw_llama_10000.json \
    --iou_threshold 0.0 \
    --detection_method flsw \
    --mode sbs

python evaluate_flsw.py --watermark kgw \
    --input_file baseline_result/flsw_main/kgw_llama_400.log \
    --data_file data/main/kgw_llama_10000.json \
    --iou_threshold 0.0 \
    --detection_method flsw \
    --mode sbs

python evaluate_flsw.py --watermark kgw \
    --input_file baseline_result/seeker_main/kgw_llama.log \
    --data_file data/main/kgw_llama_10000.json \
    --iou_threshold 0.0 \
    --detection_method seeker \
    --mode sbs

# weak but long
python evaluate_flsw.py --watermark kgw \
    --input_file baseline_result/flsw_main/kgw_llama_100.log \
    --data_file data/main/kgw_llama_10000.json \
    --iou_threshold 0.0 \
    --detection_method flsw \
    --mode wbl

python evaluate_flsw.py --watermark kgw \
    --input_file baseline_result/flsw_main/kgw_llama_200.log \
    --data_file data/main/kgw_llama_10000.json \
    --iou_threshold 0.0 \
    --detection_method flsw \
    --mode wbl

python evaluate_flsw.py --watermark kgw \
    --input_file baseline_result/seeker_main/kgw_llama.log \
    --data_file data/main/kgw_llama_10000.json \
    --iou_threshold 0.0 \
    --detection_method seeker \
    --mode wbl