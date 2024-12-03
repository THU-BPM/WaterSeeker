export PYTHONPATH="/workspace/panleyi/WaterSeeker:$PYTHONPATH"

# kgw-llama
python mix_data.py --wat_file data/main/kgw_llama_500.json \
    --non_wat_file data/non_wat/kgw_llama.json \
    --length 500 \
    --model llama

python mix_data.py --wat_file data/main/kgw_llama_2000.json \
    --non_wat_file data/non_wat/kgw_llama.json \
    --length 2000 \
    --model llama

python mix_data.py --wat_file data/main/kgw_llama_5000.json \
    --non_wat_file data/non_wat/kgw_llama.json \
    --length 5000 \
    --model llama

python mix_data.py --wat_file data/main/kgw_llama_10000.json \
    --non_wat_file data/non_wat/kgw_llama.json \
    --length 10000 \
    --model llama

# kgw-mistral
python mix_data.py --wat_file data/main/kgw_mistral_500.json \
    --non_wat_file data/non_wat/kgw_mistral.json \
    --length 500 \
    --model mistral

python mix_data.py --wat_file data/main/kgw_mistral_2000.json \
    --non_wat_file data/non_wat/kgw_mistral.json \
    --length 2000 \
    --model mistral

python mix_data.py --wat_file data/main/kgw_mistral_5000.json \
    --non_wat_file data/non_wat/kgw_mistral.json \
    --length 5000 \
    --model mistral

python mix_data.py --wat_file data/main/kgw_mistral_10000.json \
    --non_wat_file data/non_wat/kgw_mistral.json \
    --length 10000 \
    --model mistral

# aar-llama
python mix_data.py --wat_file data/main/aar_llama_10000.json \
    --non_wat_file data/non_wat/aar_llama.json \
    --length 10000 \
    --model llama

# aar-mistral
python mix_data.py --wat_file data/main/aar_mistral_10000.json \
    --non_wat_file data/non_wat/aar_mistral.json \
    --length 10000 \
    --model mistral
