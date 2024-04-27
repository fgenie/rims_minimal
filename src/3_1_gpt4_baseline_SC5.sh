python run_inference.py baseline_inference \
    --backbone gpt4turbo \
    --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
    --dataset_type ocw \
    --n 5 \
    --n_jobs 4

# # n=15 / n_jobs=4
# # takes ~ 28 hrs
python run_inference.py baseline_inference \
    --backbone gpt4turbo \
    --gsm_jslf ../dataset/gsm8K_test.jsonl \
    --dataset_type gsm \
    --n 5 \
    --n_jobs 4


#MATH
python run_inference.py baseline_inference \
	--backbone gpt4turbo \
	--gsm_jslf ../dataset/MATH/MATH-full.jsonl \
	--dataset_type math \
	--n 5 --n_jobs 4

# python run_inference.py baseline_inference \
#         --backbone gpt4turbo \
#         --gsm_jslf ../dataset/MATH/MATH-full.jsonl \
#         --dataset_type math \
#         --n 10 \
#         --n_jobs 3
