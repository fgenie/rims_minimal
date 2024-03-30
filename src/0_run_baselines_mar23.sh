# one-pot example (OCW)
# T=0, seed=777 experiment script 

# dataset_type: enum = ocw, gsm, math
# backbone: enum = chatgpt0613long, gpt4turbo --> may change to chatgpt1106, not 0613long (not required)

set -x

EXEPATH=/Users/seonils/dev/rims_minimal/src/
cd $EXEPATH


# 1 baseline run

# python run_inference.py  baseline_inference \
#                 --backbone chatgpt0613long \
#                 --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
#                 --dataset_type ocw

# python run_inference.py  baseline_inference \
#                 --backbone chatgpt0613long \
#                 --gsm_jslf ../dataset/gsm8K_test.jsonl \
#                 --dataset_type gsm

# python run_inference.py  baseline_inference \
#                 --backbone chatgpt0613long \
#                 --gsm_jslf ../dataset/gsm8K_test.jsonl \
#                 --err_idxs_f outputs/gsm8K_test_dt.gsm/chatgpt0613long/model_selection_prompts/03_23_14_35_13.jsonl.error_idxs \


# python run_inference.py baseline_inference \
#                 --backbone chatgpt0613long \
#                 --gsm_jslf ../dataset/MATH/MATH-full_pt1.jsonl \
#                 --dataset_type math

# python run_inference.py baseline_inference \
#                 --backbone chatgpt0613long \
#                 --gsm_jslf ../dataset/MATH/MATH-full_pt2.jsonl \
#                 --dataset_type math

# python run_inference.py baseline_inference \
#                 --backbone chatgpt0613long \
#                 --gsm_jslf ../dataset/MATH/MATH-full_pt3.jsonl \
#                 --dataset_type math

# python run_inference.py baseline_inference \
#                 --backbone chatgpt0613long \
#                 --gsm_jslf ../dataset/MATH/MATH-full_pt4.jsonl \
#                 --dataset_type math

# python run_inference.py baseline_inference \
#                 --backbone chatgpt0613long \
#                 --gsm_jslf ../dataset/MATH/MATH-full_pt5.jsonl \
#                 --dataset_type math






# python run_inference.py baseline_inference \
#                 --backbone chatgpt0613long \
#                 --gsm_jslf ../dataset/svamp.jsonl \
#                 --dataset_type gsm



# do with gpt-3.5-turbo-1106
python run_inference.py  baseline_inference \
                --backbone chatgpt1106 \
                --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
                --dataset_type ocw

python run_inference.py  baseline_inference \
                --backbone chatgpt1106 \
                --gsm_jslf ../dataset/gsm8K_test.jsonl \
                --dataset_type gsm


python run_inference.py baseline_inference \
                --backbone chatgpt1106 \
                --gsm_jslf ../dataset/MATH/MATH-full_pt1.jsonl \
                --dataset_type math

python run_inference.py baseline_inference \
                --backbone chatgpt1106 \
                --gsm_jslf ../dataset/MATH/MATH-full_pt2.jsonl \
                --dataset_type math

python run_inference.py baseline_inference \
                --backbone chatgpt1106 \
                --gsm_jslf ../dataset/MATH/MATH-full_pt3.jsonl \
                --dataset_type math

python run_inference.py baseline_inference \
                --backbone chatgpt1106 \
                --gsm_jslf ../dataset/MATH/MATH-full_pt4.jsonl \
                --dataset_type math

python run_inference.py baseline_inference \
                --backbone chatgpt1106 \
                --gsm_jslf ../dataset/MATH/MATH-full_pt5.jsonl \
                --dataset_type math


