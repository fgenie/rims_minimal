# one-pot example (OCW)
# T=0, seed=777 experiment script 

# dataset_type: enum = ocw, gsm, math
# backbone: enum = chatgpt0613long, gpt4turbo --> may change to chatgpt1106, not 0613long (not required)

set -x

EXEPATH=/Users/seonils/dev/rims_minimal/src/
cd $EXEPATH


# input files
GSM_INFERRED=outputs/gsm8K_test_dt.gsm/chatgpt0613long/model_selection_prompts/03_23_14_35_13.jsonl
OCW_INFERRED=outputs/ocw_course_dt.ocw/chatgpt0613long/model_selection_prompts/03_23_15_38_01.jsonl
# MATH_INFERRED=outputs/MATH-full_dt.math/chatgpt0613long/model_selection_prompts/merged.jsonl

# prompts: rims, -hint, -hint-mistakes, -hint-mistakes-attempt1
GSM_RIMS_OLD=prompt_construction_src/newer_prompts_3/rims_gsm_best.txt

GSM_RIMS=prompt_construction_src/newer_prompts_3/rims_gsm_best_newer.txt
GSM_RIMS_H=prompt_construction_src/newer_prompts_3/rims_gsm_best_newer-hint.txt
GSM_RIMS_HM=prompt_construction_src/newer_prompts_3/rims_gsm_best_newer-hint-mistakes.txt
GSM_RIMS_HMA=prompt_construction_src/newer_prompts_3/rims_gsm_best_newer-hint-mistakes-attempt1.txt

OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
OCW_RIMS_H=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint.txt
OCW_RIMS_HM=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint-mistakes.txt
OCW_RIMS_HMA=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint-mistakes-attempt1.txt


# MATH_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt
# MATH_RIMS_H=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot_-hint.txt
# MATH_RIMS_HM=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot_-hint-mistakes.txt
# MATH_RIMS_HMA=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot_-hint-mistakes-attempt1.txt

# MATH_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt1
# MATH_RIMS_H_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot_-hint.txt1
# MATH_RIMS_HM_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot_-hint-mistakes.txt1
# MATH_RIMS_HMA_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot_-hint-mistakes-attempt1.txt1

# do GSM
for PROMPT in $GSM_RIMS_OLD # $GSM_RIMS $GSM_RIMS_H $GSM_RIMS_HM $GSM_RIMS_HMA
do
    python run_inference.py rims_inference \
        --gsm_jslf ${GSM_INFERRED} \
        --dataset_type gsm \
        --prompt_f ${PROMPT}
done

# do OCW
for PROMPT in $GSM_RIMS_OLD $OCW_RIMS $OCW_RIMS_H $OCW_RIMS_HM $OCW_RIMS_HMA 
do
    python run_inference.py rims_inference \
        --gsm_jslf ${OCW_INFERRED} \
        --dataset_type ocw \
        --prompt_f ${PROMPT}
done


# # do math 
# for PROMPT in $GSM_OLD $MATH_RIMS $MATH_RIMS_H $MATH_RIMS_HM $MATH_RIMS_HMA 
# do
#     python run_inference.py rims_inference \
#         --gsm_jslf ${MATH_INFERRED} \
#         --dataset_type math \
#         --prompt_f ${PROMPT}
# done




python run_inference.py baseline_inference \
                --backbone chatgpt0613long \
                --gsm_jslf ../dataset/MATH/MATH-full_pt3.jsonl \
                --dataset_type math

python run_inference.py baseline_inference \
                --backbone chatgpt0613long \
                --gsm_jslf ../dataset/MATH/MATH-full_pt4.jsonl \
                --dataset_type math

python run_inference.py baseline_inference \
                --backbone chatgpt0613long \
                --gsm_jslf ../dataset/MATH/MATH-full_pt5.jsonl \
                --dataset_type math
