# python run_inference.py baseline_inference \
#     --backbone chatgpt1106 \
#     --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
#     --dataset_type ocw \
#     --n 10 \
#     --n_jobs 3

# # n=15 / n_jobs=4
# # takes ~ 28 hrs
# python run_inference.py baseline_inference \
#     --backbone chatgpt1106 \
#     --gsm_jslf ../dataset/gsm8K_test.jsonl \
#     --dataset_type gsm \
#     --n 15 \
#     --n_jobs 2


#MATH pt
for i in 1 2 3; do
    python run_inference.py baseline_inference \
        --backbone chatgpt1106 \
        --gsm_jslf ../dataset/MATH/MATH-full_pt${i}.jsonl \
        --dataset_type math \
        --n 15 \
        --n_jobs 6
done
