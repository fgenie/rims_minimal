# MATH_RIMS_SHORT=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/rims_math_p2c-cot.pal-p2c.txt
# MATH_RIMS_SHORT_REV=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/rims_math_pal-p2c.p2c-cot.txt
# MATH_RIMS_SHORT1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/rims_math_p2c-cot.pal-p2c.txt1
# MATH_RIMS_SHORT1_REV=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/rims_math_pal-p2c.p2c-cot.txt1
# MATH_RIMS_REORD1_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/rims_math_pal-cot.p2c-cot.pal-p2c__.txt1
# MATH_RIMS_REORD1_2=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/rims_math_pal-p2c.pal-cot.p2c-cot__.txt1

MATH_T0_BASE=outputs/0_final_results/greedy_decoded_results/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n1_baseline.jsonl



# # math
# # for PROMPT in $MATH_RIMS_SHORT1 $MATH_RIMS_SHORT1_REV $MATH_RIMS_REORD1_1 $MATH_RIMS_REORD1_2;do
# for PROMPT in $MATH_RIMS_SHORT $MATH_RIMS_SHORT_REV $MATH_RIMS_SHORT1 $MATH_RIMS_SHORT1_REV $MATH_RIMS_REORD1_1 $MATH_RIMS_REORD1_2;do
#     python run_inference.py rims_inference \
#         --backbone chatgpt1106 \
#         --gsm_jslf ${MATH_T0_BASE} \
#         --dataset_type math \
#         --prompt_f ${PROMPT} \
#         --n_jobs 16
# done
# python run_evaluation_new.py --ptn "outputs/0_final_results/greedy_decoded_results/MATH-full_dt.math/chatgpt1106/**/*.jsonl" --eval_type math --outf 5_math_newprompts_results_with_abl.txt




OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/from_ocw_modif/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
OCW_RIMS1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/from_ocw_modif/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt
OCW_RIMS_MODIF=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/from_ocw_modif/rims_ocw_pal-p2c.pal-cot__.txt
OCW_RIMS1_MODIF=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/from_ocw_modif/rims_ocw_pal-p2c.cot-p2c__.txt



# for PROMPT in $OCW_RIMS $OCW_RIMS1 $OCW_RIMS_MODIF $OCW_RIMS1_MODIF;do
for PROMPT in $OCW_RIMS1_MODIF;do
    python run_inference.py rims_inference \
        --backbone chatgpt1106 \
        --gsm_jslf ${MATH_T0_BASE} \
        --dataset_type math \
        --prompt_f ${PROMPT} \
        --n_jobs 16
done
python run_evaluation_new.py --ptn "outputs/0_final_results/greedy_decoded_results/MATH-full_dt.math/chatgpt1106/rims_ocw_pal-p2c.cot-p2c__.txt/*.jsonl" --eval_type math --outf 5_ocw_tweak_math_perf_greedy.txt






#-------------------- below: SC5 ------------------- #


# MATHRIMS_A1=prompt_construction_src/ocw_math_prep_rims_prompt/prompts/tweaks/top2/rims_math_p2c-cot.pal-p2c.txt
# MATHRIMS_A2=prompt_construction_src/ocw_math_prep_rims_prompt/prompts/tweaks/top2/rims_math_p2c-cot.pal-p2c.txt1

# rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
# rims_ocw_pal-p2c.cot-p2c__.txt

# for PROMPT in $MATHRIMS_A1 $MATHRIMS_A2; do


# for PROMPT in $OCW_RIMS_MODIF $OCW_RIMS1_MODIF; do
#     python run_inference.py rims_inference \
#                 --backbone chatgpt1106 \
#                 --gsm_jslf outputs/0_final_results/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n5_baseline.jsonl \
#                 --n 5 \
#                 --dataset_type math \
#                 --n_jobs 8 \
#                 --temperature 0.5 \
#                 --prompt_f $PROMPT
# done
# python run_evaluation_new_n.py --ptn "outputs/0_final_results/MATH-full_dt.math/chatgpt1106/rims_ocw_pal-p2c*.txt/n5_rims_T*.jsonl" --eval_type math

# for PROMPT in $OCW_RIMS $OCW_RIMS1; do
#     python run_inference.py rims_inference \
#                 --backbone chatgpt1106 \
#                 --gsm_jslf outputs/0_final_results/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n5_baseline.jsonl \
#                 --n 5 \
#                 --dataset_type math \
#                 --n_jobs 8 \
#                 --temperature 0.5 \
#                 --prompt_f $PROMPT
# done

# python run_evaluation_new_n.py --ptn "outputs/0_final_results/MATH-full_dt.math/chatgpt1106/rims_ocw*.txt/n5_rims_T*.jsonl" --eval_type math
