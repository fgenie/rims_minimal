
# # NEED TO DO on FIXED + error RIMS and merge
# # see 2_2_leftovers_rims_after_merged.sh

# # rims prompts
# OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
# OCW_RIMS1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt

# GSM_RIMS=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_newer_best_p2c2cot.pal2p2c.pal2cot.txt # looks buggy
# GSM_RIMS1=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_cot2p2c.pal2cot.pal2p2c.txt
# GSM_RIMS2=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt

# MATH_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt
# MATH_RIMS1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt1







# for T in 0.2 0.5; do
#     for PROMPT in $OCW_RIMS $OCW_RIMS1; do
#         python run_inference.py rims_inference \
#                     --backbone chatgpt1106 \
#                     --gsm_jslf outputs/n5_baseline_dt.ocw/chatgpt1106/model_selection_prompts/n5_baseline_picked.jsonl \
#                     --n 5 \
#                     --dataset_type ocw \
#                     --n_jobs 8 \
#                     --temperature $T \
#                     --prompt_f $PROMPT

#         python run_inference.py rims_inference \
#                     --backbone chatgpt1106 \
#                     --gsm_jslf outputs/n10_baseline_dt.ocw/chatgpt1106/model_selection_prompts/n10_baseline_picked.jsonl \
#                     --n 10 \
#                     --dataset_type ocw \
#                     --n_jobs 8 \
#                     --temperature $T \
#                     --prompt_f $PROMPT
#     done
# done

# for T in 0.2 0.5; do
#     for PROMPT in $GSM_RIMS $GSM_RIMS1 $GSM_RIMS2; do
#         python run_inference.py rims_inference \
#                     --backbone chatgpt1106 \
#                     --gsm_jslf outputs/n15_baseline_dt.gsm/chatgpt1106/model_selection_prompts/n15_baseline_picked.jsonl \
#                     --n 15 \
#                     --dataset_type gsm \
#                     --n_jobs 8 \
#                     --temperature $T \
#                     --prompt_f $PROMPT
#     done
# done

# for T in 0.2 0.5; do
#     for PROMPT in $MATH_RIMS $MATH_RIMS1; do
#         python run_inference.py rims_inference \
#                     --backbone chatgpt1106 \
#                     --gsm_jslf outputs/n5_baseline_dt.math/chatgpt1106/model_selection_prompts/n5_baseline_picked.jsonl \
#                     --n 5 \
#                     --dataset_type math \
#                     --n_jobs 8 \
#                     --temperature $T \
#                     --prompt_f $PROMPT
#     done
# done



# LOST_OCW5=/Users/seonils/dev/rims_minimal/src/outputs/ocw_course_dt.ocw/chatgpt1106/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt/n5_rims_T0.5.jsonl
# OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
# python run_inference.py rims_inference \
#     --backbone chatgpt1106 \
#     --gsm_jslf $LOST_OCW5 \
#     --dataset_typge ocw \
#     --temperature 0.2 \
#     --prompt_f $OCW_RIMS \
#     --n 5 \
#     --n_jobs 12





### math additional tuning!

# # original math prompt
# MATH_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt
# MATH_RIMS1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt1

# T_0508_base=outputs/0_final_results/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n5_baseline_0.5_0.8.jsonl
# T05_fixed_base=outputs/0_final_results/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n5_baseline_0.5only.jsonl

# for PROMPT in $MATH_RIMS $MATH_RIMS1; do
#     python run_inference.py rims_inference \
#                 --backbone chatgpt1106 \
#                 --gsm_jslf $T05_fixed_base  \
#                 --n 5 \
#                 --dataset_type math \
#                 --n_jobs 8 \
#                 --temperature 0.5 \
#                 --prompt_f $PROMPT
# done



T05_fixed_base=outputs/0_final_results/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n5_baseline_0.5only.jsonl

# math prompt modified (truncated, selected)
MATHRIMS_A1=prompt_construction_src/ocw_math_prep_rims_prompt/prompts/tweaks/top2_failed/rims_math_p2c-cot.pal-p2c.txt
MATHRIMS_A2=prompt_construction_src/ocw_math_prep_rims_prompt/prompts/tweaks/top2_failed/rims_math_p2c-cot.pal-p2c.txt1

# ocw prompt modified, original, selected
OCW_RIMS1_MODIF=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/from_ocw_modif/rims_ocw_pal-p2c.cot-p2c__.txt
OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/from_ocw_modif/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt

for PROMPT in $MATHRIMS_A1 $MATHRIMS_A2 $OCW_RIMS1_MODIF $OCW_RIMS; do
    python run_inference.py rims_inference \
                --backbone chatgpt1106 \
                --gsm_jslf $T05_fixed_base  \
                --n 5 \
                --dataset_type math \
                --n_jobs 8 \
                --temperature 0.5 \
                --prompt_f $PROMPT
done

python run_evaluation_new_n.py --ptn "outputs/0_final_results/MATH-full_dt.math/chatgpt1106/**/n5_rims_T0.5*.jsonl" --eval_type math --outf 5_0.5unified_otherprompts.txt
