NJOBS=$1 # network bound job: larger better for faster results


# llm
MODEL_NAME=microsoft/Phi-3-mini-128k-instruct

# baseline n=1
CMD="python run_inference.py baseline_inference \
    --backbone $MODEL_NAME \
    --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
    --dataset_type ocw \
    --n_jobs $NJOBS \
    --out_suffix user_as_sys"
echo $CMD
$CMD

BASELINE_RESULT_JSL_OCW=outputs/ocw_course_dt.ocw/microsoft/Phi-3-mini-128k-instruct/model_selection_prompts/n1_baseline_T0.0_user_as_sys.jsonl

python run_evaluation_new.py --ptn "$BASELINE_RESULT_JSL_OCW" --eval_type ocw --eval_indiv_and_overlap --outf n1_indiv.txt




# baseline n=1
CMD="python run_inference.py baseline_inference \
    --backbone $MODEL_NAME \
    --gsm_jslf ../dataset/gsm8K_test.jsonl \
    --dataset_type gsm \
    --n_jobs $NJOBS \
    --out_suffix user_as_sys"
echo $CMD
$CMD

BASELINE_RESULT_JSL_GSM=outputs/gsm8K_test_dt.gsm/microsoft/Phi-3-mini-128k-instruct/model_selection_prompts/n1_baseline_T0.0_user_as_sys.jsonl

python run_evaluation_new.py --ptn "$BASELINE_RESULT_JSL_GSM" --eval_type gsm --eval_indiv_and_overlap --outf n1_indiv.txt




# # baseline n=1
# CMD="python run_inference.py baseline_inference \
#     --backbone $MODEL_NAME \
#     --gsm_jslf ../dataset/MATH/MATH-full.jsonl \
#     --dataset_type math \
#     --n_jobs $NJOBS \
#     --out_suffix user_as_sys"
# echo $CMD
# $CMD

# BASELINE_RESULT_JSL_MATH=outputs/MATH-full_dt.math/microsoft/Phi-3-mini-128k-instruct/model_selection_prompts/n1_baseline_T0.0_user_as_sys.jsonl

# python run_evaluation_new.py --ptn "$BASELINE_RESULT_JSL_MATH" --eval_type math --eval_indiv_and_overlap --outf n1_indiv.txt








# # rims prompt --> not debugged
# OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
# OCW_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt

# # rims n=1
# for OCWPROMPT in $OCW_RIMS $OCW_RIMS_1; do
#     CMD="python run_inference.py rims_inference \
#         --backbone $MODEL_NAME \
#         --gsm_jslf /home/llm4a/Project/rims_minimal/src/outputs_phi/ocw_course_dt.ocw/Phi-3-mini-128k-instruct/model_selection_prompts/n1_baseline_T0.5_0.8_last.jsonl \
#         --dataset_type ocw \
#         --prompt_f $OCWPROMPT \
#         --n 1 --n_jobs $NJOBS"
#     echo $CMD
#     $CMD
# done