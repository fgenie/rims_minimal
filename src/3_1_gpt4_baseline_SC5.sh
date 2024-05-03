# fixing the results of gpt4-already-run
GSMBASE=outputs_dgx_gpt4turbo/gsm8K_test_dt.gsm/gpt4turbo/model_selection_prompts/n5_baseline.jsonl
# MATHBASE=outputs_dgx_gpt4turbo/MATH-full_dt.math/gpt4turbo/model_selection_prompts/n5_baseline.jsonl # only 300 rows done.
OCWBASE=outputs_dgx_gpt4turbo/ocw_course_dt.ocw/gpt4turbo/model_selection_prompts/n5_baseline.jsonl

python run_inference.py baseline_inference \
    --backbone gpt4turbo \
    --gsm_jslf $OCWBASE \
    --dataset_type ocw \
    --n 5 \
    --n_jobs 6

# # n=15 / n_jobs=4
# # takes ~ 28 hrs
python run_inference.py baseline_inference \
    --backbone gpt4turbo \
    --gsm_jslf $GSMBASE \
    --dataset_type gsm \
    --n 5 \
    --n_jobs 6


#MATH
python run_inference.py baseline_inference \
	--backbone gpt4turbo \
	--gsm_jslf ../dataset/MATH/MATH-full.jsonl \
	--dataset_type math \
	--n 5 --n_jobs 3
