# rims prompts
OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
OCW_RIMS1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt

GSM_RIMS=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_newer_best_p2c2cot.pal2p2c.pal2cot.txt # looks buggy
GSM_RIMS1=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_cot2p2c.pal2cot.pal2p2c.txt
GSM_RIMS2=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt

MATH_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt
MATH_RIMS1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt1

# rims to run
# chatgpt

# OCW n=15, | 5 + 10 leftovers




# MATH n=5 // 10, 15 (yet)
MATH5baseline=outputs/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n5_baseline.jsonl
# MATH5baseline=/Users/seonils/dev/rims_minimal/src/outputs/err_n5_baseline_dt.math/chatgpt1106/model_selection_prompts/n5_baseline.jsonl
for PROMPT in $MATH_RIMS1; do
    python run_inference.py rims_inference \
        --backbone chatgpt1106 \
        --gsm_jslf $MATH5baseline \
        --dataset_type math \
        --temperature 0.2 \
        --prompt_f $PROMPT \
        --n 5 \
        --n_jobs 16
done


# MATH5baseline=outputs/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n5_baseline.jsonl

PTNRI=outputs/MATH-full_dt.math/chatgpt1106/**/n5_rims_T0.2.jsonl
PTNBL=$MATH5baseline
for PTN in $PTNBL $PTNRI; do
    python run_evaluation_new_n.py \
        --ptn $PTN \
        --eval_type math
done



for PROMPT in $MATH_RIMS $MATH_RIMS1; do
    python run_inference.py rims_inference \
        --backbone chatgpt1106 \
        --gsm_jslf $MATH5baseline \
        --dataset_type math \
        --temperature 0.5 \
        --prompt_f $PROMPT \
        --n 5 \
        --n_jobs 16
done


PTNRI=outputs/MATH-full_dt.math/chatgpt1106/**/n5_rims_T0.5.jsonl
for PTN in $PTNRI; do
    python run_evaluation_new_n.py \
        --ptn $PTN \
        --eval_type math
done



# GSM n=5 | 10 + 15 leftovers















#-----
# # gpt4turbo
# # OCW, MATH, GSM n=5 + leftovers

# GSM5baseline_4=outputs_dgx_gpt4turbo/gsm8K_test_dt.gsm/gpt4turbo/model_selection_prompts/merged_n5_baseline.jsonl
# OCW5baseline_4=outputs_dgx_gpt4turbo/ocw_course_dt.ocw/gpt4turbo/model_selection_prompts/merged_n5_baseline.jsonl
# MATH4baseline_4_proxy=outputs_dgx_gpt4turbo/MATH-full_dt.math/gpt4turbo/model_selection_prompts/merged_n5_baseline.jsonl
# ##n=5, ocw
# for T in 0.2 0.5; do
#     for PROMPT in $OCW_RIMS $OCW_RIMS1; do
#         python run_inference.py rims_inference \
#             --backbone gpt4turbo \
#             --gsm_jslf $OCW5baseline_4 \
#             --dataset_type ocw \
#             --temperature $T \
#             --prompt_f $PROMPT \
#             --n 5 \
#             --n_jobs 8
#     done
# done

# ##n=5, gsm
# for T in 0.2 0.5; do
#     for PROMPT in $GSM_RIMS $GSM_RIMS1 $GSM_RIMS2; do
#         python run_inference.py rims_inference \
#             --backbone gpt4turbo \
#             --gsm_jslf $GSM5baseline_4 \
#             --dataset_type gsm \
#             --temperature $T \
#             --prompt_f $PROMPT \
#             --n 5 \
#             --n_jobs 8
#     done
# done

# ##n=5, math proxy (314 over 5000)
# for T in 0.2 0.5; do
#     for PROMPT in $MATH_RIMS $MATH_RIMS1; do
#         python run_inference.py rims_inference \
#             --backbone gpt4turbo \
#             --gsm_jslf $MATH4baseline_4_proxy \
#             --dataset_type math \
#             --temperature $T \
#             --prompt_f $PROMPT \
#             --n 5 \
#             --n_jobs 8
#     done
# done







# PTNRI=outputs_dgx_gpt4turbo/gsm8K_test_dt.gsm/gpt4turbo/**/n5_rims_T*.jsonl
# PTNBL=$GSM5baseline_4
# for PTN in $PTNBL $PTNRI; do
#     python run_evaluation_new_n.py \
#         --ptn $PTN \
#         --eval_type gsm
# done


# PTNRI=outputs_dgx_gpt4turbo/ocw_course_dt.ocw/gpt4turbo/**/n5_rims_T*.jsonl
# PTNBL=$OCW5baseline_4
# for PTN in $PTNBL $PTNRI; do
#     python run_evaluation_new_n.py \
#         --ptn $PTN \
#         --eval_type ocw
# done



# PTNRI=outputs_dgx_gpt4turbo/MATH-full_dt.math/gpt4turbo/**/n5_rims_T*.jsonl
# PTNBL=$MATH5baseline_4
# for PTN in $PTNBL $PTNRI; do
#     python run_evaluation_new_n.py \
#         --ptn $PTN \
#         --eval_type math
# done
