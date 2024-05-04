for i in {1..25}; do
    python run_inference.py baseline_inference \
        --backbone chatgpt1106 \
        --gsm_jslf ../dataset/MATH/MATH-full_pt${i}.jsonl \
        --dataset_type math \
        --n 10 \
        --n_jobs 4
done
