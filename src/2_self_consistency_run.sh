
set -x

# EXEPATH=/Users/seonils/dev/rims_minimal/src/
# cd $EXEPATH

# TPM 300K 기준 n_jobs

# n=15 / n_jobs=2
# n=5 / n_jobs=5
# python run_inference.py baseline_inference \
#     --backbone chatgpt1106 \
#     --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
#     --dataset_type ocw \
#     --n 5 \
#     --n_jobs 5


OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
OCW_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt
OCW_BASELINE_DONE=outputs/ocw_course_dt.ocw/chatgpt1106/model_selection_prompts/n5_baseline.jsonl

for PROMPT in $OCW_RIMS $OCW_RIMS_1; do
    python run_inference.py rims_inference \
        --backbone chatgpt1106 \
        --gsm_jslf $OCW_BASELINE_DONE \
        --dataset_type ocw \
        --temperature 0.5 \
        --prompt_f $PROMPT \
        --n 5 \
        --n_jobs 16
done



# PTNBL=outputs/ocw_course_dt.ocw/chatgpt1106/**/n5_baseline.jsonl
PTNRI=outputs/ocw_course_dt.ocw/chatgpt1106/**/n5_rims_T*.jsonl

# for PTN in $PTNBL $PTNRI; do
for PTN in $PTNRI; do
    python run_evaluation_new_n.py \
        --ptn $PTN \
        --eval_type ocw
done
