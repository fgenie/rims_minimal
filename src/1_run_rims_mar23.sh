# one-pot example (OCW)
# T=0, seed=777 experiment script

# dataset_type: enum = ocw, gsm, math
# backbone: enum = chatgpt0613long, gpt4turbo --> may change to chatgpt1106, not 0613long (not required)

set -x

EXEPATH=/Users/seonils/dev/rims_minimal/src/
cd $EXEPATH


# input files
GSM_INFERRED=outputs/gsm8K_test_dt.gsm/chatgpt0613long/model_selection_prompts/03_25_01_21_45.jsonl
OCW_INFERRED=outputs/ocw_course_dt.ocw/chatgpt0613long/model_selection_prompts/03_23_15_38_01.jsonl
MATH_INFERRED=outputs/MATH-full_dt.math/chatgpt0613long/model_selection_prompts/merged.jsonl



GSM_INFERRED1106=outputs/gsm8K_test_dt.gsm/chatgpt1106/model_selection_prompts/03_30_02_18_36.jsonl
OCW_INFERRED1106=outputs/ocw_course_dt.ocw/chatgpt1106/model_selection_prompts/03_30_01_27_09.jsonl
MATH_INFERRED1106=outputs/MATH-full_dt.math/chatgpt1106/model_selection_prompts/merged.jsonl

# prompts: rims, -hint, -hint-mistakes, -hint-mistakes-attempt1
GSM_RIMS_OLD=prompt_construction_src/newer_prompts_3/rims_gsm_best.txt

GSM_RIMS=prompt_construction_src/newer_prompts_3/rims_gsm_best_newer.txt
GSM_RIMS_H=prompt_construction_src/newer_prompts_3/rims_gsm_best_newer-hint.txt
GSM_RIMS_HM=prompt_construction_src/newer_prompts_3/rims_gsm_best_newer-hint-mistakes.txt
GSM_RIMS_HMA=prompt_construction_src/newer_prompts_3/rims_gsm_best_newer-hint-mistakes-attempt1.txt


GSM_RIMS_RW=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_newer_best_p2c2cot.pal2p2c.pal2cot.txt
GSM_RIMS_RW_1=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_cot2p2c.pal2cot.pal2p2c.txt
GSM_RIMS_RW_2=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt



OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
OCW_RIMS_H=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint.txt
OCW_RIMS_HM=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint-mistakes.txt
OCW_RIMS_HMA=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot_-hint-mistakes-attempt1.txt

OCW_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt
OCW_RIMS_H_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c_-hint.txt
OCW_RIMS_HM_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c_-hint-mistakes.txt
OCW_RIMS_HMA_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c_-hint-mistakes-attempt1.txt

MATH_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt
MATH_RIMS_H=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot_-hint.txt
MATH_RIMS_HM=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot_-hint-mistakes.txt
MATH_RIMS_HMA=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot_-hint-mistakes-attempt1.txt

MATH_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt1
MATH_RIMS_H_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot_-hint.txt1
MATH_RIMS_HM_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot_-hint-mistakes.txt1
MATH_RIMS_HMA_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot_-hint-mistakes-attempt1.txt1


# # do GSM
# for PROMPT in $GSM_RIMS_OLD $GSM_RIMS $GSM_RIMS_H $GSM_RIMS_HM $GSM_RIMS_HMA
# do
#     python run_inference.py rims_inference \
#         --gsm_jslf ${GSM_INFERRED} \
#         --dataset_type gsm \
#         --prompt_f ${PROMPT}
# done



# # do OCW # turned out bad (ocw_results.txt)
# for PROMPT in $GSM_OLD $OCW_RIMS $OCW_RIMS_H $OCW_RIMS_HM $OCW_RIMS_HMA
# do
#     python run_inference.py rims_inference \
#         --gsm_jslf ${OCW_INFERRED} \
#         --dataset_type ocw \
#         --prompt_f ${PROMPT}
# done


# for PROMPT in $OCW_RIMS_1 $OCW_RIMS_H_1 $OCW_RIMS_HM_1 $OCW_RIMS_HMA_1
# do
#     python run_inference.py rims_inference \
#         --gsm_jslf ${OCW_INFERRED} \
#         --dataset_type ocw \
#         --prompt_f ${PROMPT}
# done




# # do math
# for PROMPT in $GSM_OLD $MATH_RIMS $MATH_RIMS_H $MATH_RIMS_HM $MATH_RIMS_HMA
# do
#     python run_inference.py rims_inference \
#         --gsm_jslf ${MATH_INFERRED} \
#         --dataset_type math \
#         --prompt_f ${PROMPT}
# done




# 1106 infer

# # ocw

# for PROMPT in $GSM_RIMS_OLD $OCW_RIMS $OCW_RIMS_H $OCW_RIMS_HM $OCW_RIMS_HMA
# do
#     python run_inference.py rims_inference \
#         --gsm_jslf ${OCW_INFERRED1106} \
#         --dataset_type ocw \
#         --prompt_f ${PROMPT} \
#         --backbone chatgpt1106
# done


# for PROMPT in $OCW_RIMS $OCW_RIMS_H $OCW_RIMS_HM $OCW_RIMS_HMA
# do
#     python run_inference.py rims_inference \
#         --gsm_jslf ${OCW_INFERRED1106} \
#         --dataset_type ocw \
#         --prompt_f ${PROMPT} \
#         --backbone chatgpt1106
# done


# # gsm
# for PROMPT in $GSM_RIMS_OLD $GSM_RIMS $GSM_RIMS_H $GSM_RIMS_HM $GSM_RIMS_HMA
# do
#     python run_inference.py rims_inference \
#         --gsm_jslf ${GSM_INFERRED1106} \
#         --dataset_type gsm \
#         --prompt_f ${PROMPT} \
#         --backbone chatgpt1106
# done







# # gsm renewed prompts (p2c prepped by chatgpt to make it more looklike p2c)
# for PROMPT in $GSM_RIMS_RW $GSM_RIMS_RW_1 $GSM_RIMS_RW_2
# do
#     python run_inference.py rims_inference \
#         --backbone chatgpt1106 \
#         --gsm_jslf ${GSM_INFERRED1106} \
#         --dataset_type gsm \
#         --prompt_f ${PROMPT}
# done
# python run_evaluation_new.py --ptn "outputs/gsm8K_test_dt.gsm/chatgpt1106/rewrote.p2c_gsm*/*jsonl" --eval_type gsm --outf gsm_results_renewed_012.txt



# do math
# python run_evaluation_new.py --ptn "outputs/MATH-full_dt.math/chatgpt1106/model_selection_prompts/merged.jsonl" --eval_type math --outf math1106_baseline.txt --eval_indiv_and_overlap
# for PROMPT in $MATH_RIMS $MATH_RIMS_H $MATH_RIMS_HM $MATH_RIMS_HMA
# for PROMPT in $GSM_RIMS_OLD $MATH_RIMS_1 $MATH_RIMS_H_1 $MATH_RIMS_HM_1 $MATH_RIMS_HMA_1
# do
#     python run_inference.py rims_inference \
#         --backbone chatgpt1106 \
#         --gsm_jslf ${MATH_INFERRED1106} \
#         --dataset_type math \
#         --prompt_f ${PROMPT}
# done
# python run_evaluation_new.py --ptn "outputs/MATH-full_dt.math/chatgpt1106/*/*jsonl" --eval_type math --outf math1106_results.txt






# gpt4turbo

# # timedout rows
# OCW4_TIMEDOUT=outputs/ocw_course_dt.ocw/gpt4turbo/model_selection_prompts/timedout.jsonl
# MATH4_TIMEDOUT=outputs/MATH-full_dt.math/gpt4turbo/model_selection_prompts/timedout.jsonl
# python run_inference.py  baseline_inference \
#                 --backbone gpt4turbo \
#                 --gsm_jslf ${OCW4_TIMEDOUT} \
#                 --dataset_type ocw

# python run_inference.py  baseline_inference \
#                 --backbone gpt4turbo \
#                 --gsm_jslf ${MATH4_TIMEDOUT} \
#                 --dataset_type math


# outputs/timedout_dt.ocw/gpt4turbo/model_selection_prompts/04_04_11_30_43.jsonl
# outputs/timedout_dt.math/gpt4turbo/model_selection_prompts/04_04_11_42_27.jsonl
GSM4_INFERRED=outputs/gsm8K_test_dt.gsm/gpt4turbo/model_selection_prompts/04_03_21_43_44.jsonl
OCW4_INFERRED=outputs/ocw_course_dt.ocw/gpt4turbo/model_selection_prompts/04_03_00_10_55.jsonl
MATH4_INFERRED=outputs/MATH-full_dt.math/gpt4turbo/model_selection_prompts/merged.jsonl


# ocw
for PROMPT in $GSM_RIMS_OLD $OCW_RIMS $OCW_RIMS_1
do
    python run_inference.py rims_inference \
        --gsm_jslf ${OCW4_INFERRED} \
        --dataset_type ocw \
        --prompt_f ${PROMPT} \
        --backbone gpt4turbo
done


# gsm
for PROMPT in $GSM_RIMS_OLD $GSM_RIMS $GSM_RIMS_RW $GSM_RIMS_RW_1 $GSM_RIMS_RW_2
do
    python run_inference.py rims_inference \
        --gsm_jslf ${GSM4_INFERRED} \
        --dataset_type gsm \
        --prompt_f ${PROMPT} \
        --backbone gpt4turbo
done


# math
for PROMPT in $GSM_RIMS_OLD $MATH_RIMS $MATH_RIMS_1
do
    python run_inference.py rims_inference \
        --backbone gpt4turbo \
        --gsm_jslf ${MATH4_INFERRED} \
        --dataset_type math \
        --prompt_f ${PROMPT}
done


# evals

# baselines
# python run_evaluation_new.py --ptn "outputs/MATH-full_dt.math/gpt4turbo/model_selection_prompts/merged.jsonl" --eval_type math --outf math4_baseline.txt --eval_indiv_and_overlap
# python run_evaluation_new.py --ptn "outputs/gsm8K_test_dt.gsm/gpt4turbo/model_selection_prompts/04_03_21_43_44.jsonl" --eval_type gsm --outf gsm4_baseline.txt --eval_indiv_and_overlap
# python run_evaluation_new.py --ptn "outputs/ocw_course_dt.ocw/gpt4turbo/model_selection_prompts/04_03_00_10_55.jsonl" --eval_type ocw --outf ocw4_baseline.txt --eval_indiv_and_overlap

# rims_results
# python run_evaluation_new.py --ptn "outputs/MATH-full_dt.math/gpt4turbo/*/*jsonl" --eval_type math --outf math4_results.txt
# python run_evaluation_new.py --ptn "outputs/gsm8K_test_dt.gsm/gpt4turbo/*/*jsonl" --eval_type gsm --outf gsm4_results.txt
# python run_evaluation_new.py --ptn "outputs/ocw_course_dt.ocw/gpt4turbo/*/*jsonl" --eval_type ocw --outf ocw4_results.txt


python run_selection_feedback_analyses.py --ptn "outputs/MATH-full_dt.math/gpt4turbo/*/*jsonl" --eval_type math --outf select_feedback_analyses_gpt41106.txt
python run_selection_feedback_analyses.py --ptn "outputs/gsm8K_test_dt.gsm/gpt4turbo/*/*jsonl" --eval_type gsm --outf select_feedback_analyses_gpt41106.txt
python run_selection_feedback_analyses.py --ptn "outputs/ocw_course_dt.ocw/gpt4turbo/*/*jsonl" --eval_type ocw --outf select_feedback_analyses_gpt41106.txt

python run_selection_feedback_analyses.py --ptn "outputs/MATH-full_dt.math/chatgpt1106/*/*jsonl" --eval_type math --outf select_feedback_analyses_chatgpt1106.txt
python run_selection_feedback_analyses.py --ptn "outputs/gsm8K_test_dt.gsm/chatgpt1106/*/*jsonl" --eval_type gsm --outf select_feedback_analyses_chatgpt1106.txt
python run_selection_feedback_analyses.py --ptn "outputs/ocw_course_dt.ocw/chatgpt1106/*/*jsonl" --eval_type ocw --outf select_feedback_analyses_chatgpt1106.txt
