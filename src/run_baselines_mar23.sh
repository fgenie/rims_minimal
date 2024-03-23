# one-pot example (OCW)
# T=0, seed=777 experiment script 

# dataset_type: enum = ocw, gsm, math
# backbone: enum = chatgpt0613long, gpt4turbo --> may change to chatgpt1106, not 0613long (not required)

set -x

EXEPATH=/Users/seonils/dev/rims_minimal/src/
cd $EXEPATH


# 1 baseline run
python run_inference.py  baseline_inference \
                --backbone chatgpt0613long \
                --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
                --dataset_type ocw

python run_inference.py  baseline_inference \
                --backbone chatgpt0613long \
                --gsm_jslf ../dataset/gsm8K_test.jsonl \
                --dataset_type gsm


python run_inference.py basline_inference \
                --backbone chatgpt0613long \
                --gsm_jslf ../dataset/MATH-full.jsonl \
                --dataset_type math

python run_inference.py basline_inference \
                --backbone chatgpt0613long \
                --gsm_jslf ../dataset/svamp.jsonl \
                --dataset_type gsm
