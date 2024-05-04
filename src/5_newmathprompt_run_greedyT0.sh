MATH_RIMS_SHORT=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/rims_math_p2c-cot.pal-p2c.txt
MATH_RIMS_SHORT_REV=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/rims_math_pal-p2c.p2c-cot.txt
MATH_RIMS_SHORT1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/rims_math_p2c-cot.pal-p2c.txt1
MATH_RIMS_SHORT1_REV=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/rims_math_pal-p2c.p2c-cot.txt1
MATH_RIMS_REORD1_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/rims_math_pal-cot.p2c-cot.pal-p2c__.txt1
MATH_RIMS_REORD1_2=prompt_construction_src/newer_prompts_3/math_ocw_prompts/tweaks/rims_math_pal-p2c.pal-cot.p2c-cot__.txt1

MATH_T0_BASE=outputs/0_final_results/greedy_decoded_results/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n1_baseline.jsonl

# math
for PROMPT in $MATH_RIMS_SHORT $MATH_RIMS_SHORT_REV $MATH_RIMS_SHORT1 $MATH_RIMS_SHORT1_REV $MATH_RIMS_REORD1_1 $MATH_RIMS_REORD1_2;do
    python run_inference.py rims_inference \
        --backbone chatgpt1106 \
        --gsm_jslf ${MATH_T0_BASE} \
        --dataset_type math \
        --prompt_f ${PROMPT} \
        --n_jobs 16
done

python run_evaluation_new.py --ptn "outputs/0_final_results/greedy_decoded_results/MATH-full_dt.math/chatgpt1106/**/*.jsonl" --eval_type math --outf 5_math_newprompts_results_with_abl.txt
