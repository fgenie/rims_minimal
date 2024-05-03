
WRONGLY_DONE_OCWBASE_10=outputs_dgx/ocw_course_dt.ocw/chatgpt1106/model_selection_prompts/n10_baseline.jsonl
WRONGLY_DONE_OCWBASE_5=outputs/ocw_course_dt.ocw/chatgpt1106/model_selection_prompts/n5_baseline.jsonl
# python run_inference.py baseline_inference \
#     --backbone chatgpt1106 \
#     --gsm_jslf $WRONGLY_DONE_OCWBASE_10 \
#     --dataset_type ocw \
#     --n 10 \
#     --n_jobs 4

# python run_inference.py baseline_inference \
#     --backbone chatgpt1106 \
#     --gsm_jslf $WRONGLY_DONE_OCWBASE_5 \
#     --dataset_type ocw \
#     --n 5 \
#     --n_jobs 4

FIXED_OCW_BASE_10=outputs_dgx/n10_baseline.ocw/chatgpt1106/model_selection_prompts/n10_baseline.jsonl
FIXED_OCW_BASE_5=outputs/n5_baseline_dt.ocw/chatgpt1106/model_selection_prompts/n5_baseline.jsonl

# # n=15 / n_jobs=4
# # takes ~ 28 hrs
WRONGLY_DONE_GSMBASE=outputs_dgx/gsm8K_test_dt.gsm/chatgpt1106/model_selection_prompts/n15_baseline.jsonl
# python run_inference.py baseline_inference \
#     --backbone chatgpt1106 \
#     --gsm_jslf $WRONGLY_DONE_GSMBASE \
#     --dataset_type gsm \
#     --n 15 \
#     --n_jobs 4

FIXED_GSM_BASE_15=outputs_dgx/n15_baseline.gsm/chatgpt1106/model_selection_prompts/n15_baseline.jsonl

#MATH
# does not affect rims result
WRONGLY_DONE_MATHBASE=outputs/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n5_baseline.jsonl
python run_inference.py baseline_inference \
	--backbone chatgpt1106 \
	--gsm_jslf $WRONGLY_DONE_MATHBASE \
	--dataset_type math \
	--n 5 --n_jobs 4

FIXED_MATH_BASE_5=outputs/n5_baseline_dt.math/chatgpt1106/model_selection_prompts/n5_baseline.jsonl



python run_inference.py baseline_inference \
        --backbone chatgpt1106 \
        --gsm_jslf ../dataset/MATH/MATH-full.jsonl \
        --dataset_type math \
        --n 10 --n_jobs 7

N10_MATH_BASELINE_GPT35=outputs/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n10_baseline.jsonl



MATH_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt
MATH_RIMS1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt1



for T in 0.2 0.5; do
    for PROMPT in $MATH_RIMS $MATH_RIMS1; do
        python run_inference.py rims_inference \
            --backbone chatgpt1106 \
            --gsm_jslf $N10_MATH_BASELINE_GPT35 \
            --dataset_type math \
            --temperature $T \
            --prompt_f $PROMPT \
            --n 10 \
            --n_jobs 8
    done
done






PTNRI=outputs/MATH-full_dt.math/chatgpt1106/**/n5_rims_T*.jsonl
PTNRI1=outputs/MATH-full_dt.math/chatgpt1106/**/n10_rims_T*.jsonl
PTNBL=$FIXED_MATH_BASE_5
PTNBL1=$N10_MATH_BASELINE_GPT35
PTNBL2=$FIXED_OCW_BASE_5
PTNBL3=$FIXED_OCW_BASE_10
PTNBL4=$FIXED_GSM_BASE_15

for PTN in $PTNBL $PTNRI $PTNBL1 $PTNRI1 $PTNBL2 $PTNBL3 $PTNBL4; do
    python run_evaluation_new_n.py \
        --ptn $PTN \
        --eval_type math
done
