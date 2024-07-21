set -x

python postprocess_rawouts.py process_rims --ptn "outputs/*/Meta-Llama-3-8B-Instruct/rims/rims_*/" --n 1

PTN1=outputs/*/Meta-Llama-3-8B-Instruct/rims/rims*/processed_rims.jsonl

python score_processed.py score_rims --ptn "$PTN1" --n 1
