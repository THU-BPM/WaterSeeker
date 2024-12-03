export PYTHONPATH="/workspace/panleyi/WaterSeeker:$PYTHONPATH"

# generate data for main experiment & changing N
## 会生成4个files
python -u generate_data.py --num_sample 300 \
    --wiki_start_index 0 \
    --prompt_start_index 0 \
    --watermark kgw \
    --key 33554393 \
    --model llama \
    --output_file data/main/kgw_llama.json

python -u generate_data.py --num_sample 300 \
    --wiki_start_index 0 \
    --prompt_start_index 0 \
    --watermark kgw \
    --key 4294967291 \
    --model mistral \
    --output_file data/main/kgw_mistral.json

python -u generate_data.py --num_sample 300 \
    --wiki_start_index 0 \
    --prompt_start_index 0 \
    --watermark aar \
    --seed 42 \
    --model llama \
    --output_file data/main/aar_llama.json

python -u generate_data.py --num_sample 300 \
    --wiki_start_index 0 \
    --prompt_start_index 0 \
    --watermark aar \
    --seed 42 \
    --model mistral \
    --output_file data/main/aar_mistral.json