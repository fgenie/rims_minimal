# # gpt4turbo
# GSM4=outputs_dgx_gpt4turbo/gsm8K_test_dt.gsm/gpt4turbo/model_selection_prompts/err_n5_baseline.jsonl
# MATH4=outputs_dgx_gpt4turbo/MATH-full_dt.math/gpt4turbo/model_selection_prompts/err_n5_baseline.jsonl
# OCW4=outputs_dgx_gpt4turbo/ocw_course_dt.ocw/gpt4turbo/model_selection_prompts/err_n5_baseline.jsonl

# chatgpt1106
# GSM15=outputs_dgx/gsm8K_test_dt.gsm/chatgpt1106/model_selection_prompts/err_n15_baseline.jsonl
# OCW10=outputs_dgx/ocw_course_dt.ocw/chatgpt1106/model_selection_prompts/err_n10_baseline.jsonl
# OCW5=outputs/ocw_course_dt.ocw/chatgpt1106/model_selection_prompts/err_n5_baseline.jsonl
# MATH5=outputs/MATH-full_dt.math/chatgpt1106/model_selection_prompts/err_n5_baseline.jsonl


# # gpt4turbo runs
# python run_inference.py baseline_inference \
#     --backbone gpt4turbo \
#     --gsm_jslf $GSM4 \
#     --dataset_type gsm \
#     --n 5 \
#     --n_jobs 3

# python run_inference.py baseline_inference \
#     --backbone gpt4turbo \
#     --gsm_jslf $MATH4 \
#     --dataset_type math \
#     --n 5 \
#     --n_jobs 3

# python run_inference.py baseline_inference \
#     --backbone gpt4turbo \
#     --gsm_jslf $OCW4 \
#     --dataset_type ocw \
#     --n 5 \
#     --n_jobs 3


# chatgpt1106 runs
# python run_inference.py baseline_inference \
#     --backbone chatgpt1106 \
#     --gsm_jslf $GSM15 \
#     --dataset_type gsm \
#     --n 15 \
#     --n_jobs 1

# python run_inference.py baseline_inference \
#     --backbone chatgpt1106 \
#     --gsm_jslf $OCW10 \
#     --dataset_type ocw \
#     --n 10 \
#     --n_jobs 2

# python run_inference.py baseline_inference \
#     --backbone chatgpt1106 \
#     --gsm_jslf $OCW5 \
#     --dataset_type ocw \
#     --n 5 \
#     --n_jobs 2


# python run_inference.py baseline_inference \
#     --backbone chatgpt1106 \
#     --gsm_jslf $MATH5 \
#     --dataset_type math \
#     --n 5 \
#     --n_jobs 6



ERR_IDX_10_MATH=/Users/seonils/dev/rims_minimal/src/outputs/MATH-full_dt.math/chatgpt1106/model_selection_prompts/n10_baseline.jsonl.error_idxs

python run_inference.py baseline_inference \
    --backbone chatgpt1106 \
    --gsm_jslf ../dataset/MATH/MATH-full.jsonl \
    --dataset_type math \
    --err_idxs_f $ERR_IDX_10_MATH \
    --n 10 \
    --n_jobs 4
