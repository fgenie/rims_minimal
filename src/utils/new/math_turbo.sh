# python run_inference.py  baseline_inference \
#                 --backbone chatgpt \
#                 --outdir seonil_scripts/math_result/new \
#                 --gsm_jslf /Users/seonils/dev/rims_minimal/src/seonil_scripts/math_result/chatgpt_math_baseline.jsonl \
#                 --dataset_type math 

EXEPATH=/Users/seonils/dev/rims_minimal/src/
cd $EXEPATH


python run_inference.py  baseline_inference \
                --backbone gpt4turbo \
                --outdir new/ \
                --gsm_jslf new/seed/gpt4turbo_math_.jsonl \
                --dataset_type math 

# pid = $!
# wait $pid 

# python run_inference.py rims_inference \
#                             --prompt_f prompt_construction_src/prep_rims_prompts/gsm_prompts/3_reflectonce_cot2p2c.pal2cot.pal2p2c.txt_rm_ans \
#                             --gsm_jslf /Users/seonils/dev/rims_minimal/src/seonil_scripts/math_result/gpt4turbo_math_.jsonl \
#                             --dataset_type math \
#                             --backbone chatgpt \
#                             --outdir seonil_scripts/dbg \
#                             --running_on_prev_result false
