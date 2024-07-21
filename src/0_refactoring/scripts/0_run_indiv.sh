python run_baseline.py --gsm_jslf ../../dataset/gsm8K_test.jsonl --dataset_type gsm
python run_baseline.py --gsm_jslf ../../dataset/ocw/ocw_course.jsonl --dataset_type ocw
python run_baseline.py --gsm_jslf ../../dataset/MATH/MATH-full.jsonl --dataset_type math

# How to run?
# OPENAI_API_BASE=http://localhost:8000/v1 python run_baseline.py --gsm_jslf=some_dir/gsm8K_test.jsonl --dataset_type=gsm --backbone=Meta-Llama-3-8B-Instruct
# retry error row only
# OPENAI_API_BASE=http://localhost:8000/v1 python run_baseline.py --gsm_jslf=/data/recoteam_583/joel/opensource/rims_minimal/dataset/gsm8K_test.jsonl --dataset_type=gsm --backbone=Meta-Llama-3-8B-Instruct --retry_error_in_result_file_path=outputs/gsm8K_test_dt.gsm/Meta-Llama-3-8B-Instruct/n1_baseline_raw_query_result.jsonl
