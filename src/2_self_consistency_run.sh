
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
OCW_BASELINE_5=outputs/ocw_course_dt.ocw/chatgpt1106/model_selection_prompts/n5_baseline.jsonl
OCW_BASELINE_10=outputs_dgx/ocw_course_dt.ocw/chatgpt1106/model_selection_prompts/n10_baseline.jsonl

##n=5, ocw
# for PROMPT in $OCW_RIMS $OCW_RIMS_1; do
#     python run_inference.py rims_inference \
#         --backbone chatgpt1106 \
#         --gsm_jslf $OCW_BASELINE_5 \
#         --dataset_type ocw \
#         --temperature 0.2 \
#         --prompt_f $PROMPT \
#         --n 5 \
#         --n_jobs 16
# done



##n=10, ocw
for T in 0.2 0.5; do
    for PROMPT in $OCW_RIMS $OCW_RIMS_1; do
        python run_inference.py rims_inference \
            --backbone chatgpt1106 \
            --gsm_jslf $OCW_BASELINE_10 \
            --dataset_type ocw \
            --temperature $T \
            --prompt_f $PROMPT \
            --n 10 \
            --n_jobs 8
    done
done



PTNRI=outputs_dgx/ocw_course_dt.ocw/chatgpt1106/**/n10_rims_T*.jsonl
for PTN in $PTNRI; do
    python run_evaluation_new_n.py \
        --ptn $PTN \
        --eval_type ocw
done




GSM_BASELINE_15=outputs_dgx/gsm8K_test_dt.gsm/chatgpt1106/model_selection_prompts/n15_baseline.jsonl
GSM_RIMS=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_newer_best_p2c2cot.pal2p2c.pal2cot.txt # looks buggy
GSM_RIMS1=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_cot2p2c.pal2cot.pal2p2c.txt
GSM_RIMS2=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt

#n=15 gsm
for T in 0.2 0.5; do
    for PROMPT in  $GSM_RIMS $GSM_RIMS1 $GSM_RIMS2; do
        python run_inference.py rims_inference \
            --backbone chatgpt1106 \
            --gsm_jslf $GSM_BASELINE_15 \
            --dataset_type gsm \
            --temperature $T \
            --prompt_f $PROMPT \
            --n 15 \
            --n_jobs 8
    done
done



PTNBL=outputs_dgx/gsm8K_test_dt.gsm/chatgpt1106/**/n15_baseline.jsonl
PTNRI=outputs_dgx/gsm8K_test_dt.gsm/chatgpt1106/**/n15_rims_T*.jsonl

for PTN in $PTNBL $PTNRI; do
    python run_evaluation_new_n.py \
        --ptn $PTN \
        --eval_type gsm
done
