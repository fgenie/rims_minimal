
set -x

# EXEPATH=/Users/seonils/dev/rims_minimal/src/
# cd $EXEPATH

python run_inference.py baseline_inference \
    --backbone chatgpt1106 \
    --gsm_jslf ../dataset/gsm8K_test.jsonl \
    --dataset_type gsm \
    --n 15 \
    --n_jobs 8

python run_inference.py rims_inference \
    --backbone chatgpt1106 \
    --gsm_jslf PREV_RESULT_JSL \
    --dataset_type gsm \
    --temperature 0.7 \
    --prompt_f prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt \
    --n 15 \
    --n_jobs 8
