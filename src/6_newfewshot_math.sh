

MATH_T0_BASE_FIXED=outputs/0_final_results/greedy_decoded_results/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n1_baseline_fixed.jsonl


# python run_evaluation_new.py --ptn $MATH_T0_BASE_FIXED --eval_type math --outf 6_newly_built_prompts_math.txt

NEWMATH1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/newprompts/newmath_rims_1.txt
NEWMATH2=prompt_construction_src/newer_prompts_3/math_ocw_prompts/newprompts/newmath_rims_2.txt
NEWMATH3=prompt_construction_src/newer_prompts_3/math_ocw_prompts/newprompts/newmath_rims_3.txt


# for PROMPT in $NEWMATH1 $NEWMATH2 $NEWMATH3;do
# for PROMPT in $NEWMATH2 $NEWMATH3;do
for PROMPT in $NEWMATH2;do
    python run_inference.py rims_inference \
        --backbone chatgpt1106 \
        --gsm_jslf ${MATH_T0_BASE_FIXED} \
        --dataset_type math \
        --prompt_f ${PROMPT} \
        --n_jobs 8 \
        --start_idx 4970
done

python run_evaluation_new.py --ptn "outputs/0_final_results/greedy_decoded_results/MATH-full_dt.math/chatgpt1106/**/newmath_rims_*.txt/*.jsonl" --eval_type math --outf 6_newly_built_prompts_math.txt






#-------------------- below: SC5 ------------------- #


# SC5BASE=outputs/0_final_results/MATH-full_dt.math/chatgpt1106/model_selection_prompts/modif_n5_baseline_0.5_0.8.jsonl

# for PROMPT in $NEWMATH1 $NEWMATH2 $NEWMATH3;do
#     python run_inference.py rims_inference \
#                 --backbone chatgpt1106 \
#                 --gsm_jslf $SC5BASE \
#                 --n 5 \
#                 --dataset_type math \
#                 --n_jobs 8 \
#                 --temperature 0.5 \
#                 --prompt_f $PROMPT
# done

# python run_evaluation_new.py --ptn "outputs/0_final_results/greedy_decoded_results/MATH-full_dt.math/chatgpt1106/**/*_newfewshots.jsonl" --eval_type math --outf 6_newly_built_prompts_math.txt
