# # run on dgx
# # leftover baseline ocw
python run_inference.py baseline_inference \
    --backbone chatgpt1106 \
    --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
    --dataset_type ocw \
    --n 10 \
    --n_jobs 4

# n=15 / n_jobs=4
# takes ~ 28 hrs
python run_inference.py baseline_inference \
    --backbone chatgpt1106 \
    --gsm_jslf ../dataset/gsm8K_test.jsonl \
    --dataset_type gsm \
    --n 15 \
    --n_jobs 3
