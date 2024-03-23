# one-pot example (OCW)
# T=0, seed=777 experiment script 

# dataset_type: enum = ocw, gsm, svamp, math
# backbone: enum = chatgpt0613long, gpt4turbo

set -x

EXEPATH=/Users/seonils/dev/rims_minimal/src/
cd $EXEPATH

V3PROMPT=prompt_construction_src/prep_rims_prompts/gsm_prompts/3_reflectonce_p2c2cot.pal2p2c.pal2cot.txt_rm_ans

# 1 baseline run
python run_inference.py  baseline_inference \
                --backbone chatgpt0613long \
                --outdir $BASELINE_RESULT_DIR \
                --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
                --dataset_type ocw



# 2 rims run (v3 prompt)
python run_inference.py rims_inference \
                            --prompt_f $V3PROMPT \
                            --gsm_jslf $BASELINE_RESULT_DIR/chatgpt0613long_model_selection3_ocw.jsonl \
                            --dataset_type ocw \
                            --backbone chatgpt0613long \
                            --outdir $RIMS_RESULT_DIR 
# rims ablation run (v3 prompt)                            
python run_inference.py rims_inference \
                            --prompt_f $V3PROMPT \
                            --gsm_jslf $BASELINE_RESULT_DIR/chatgpt0613long_model_selection3_ocw.jsonl \
                            --dataset_type ocw \
                            --backbone chatgpt0613long \
                            --outdir $ABL_RESULT_DIR 

mkdir -p $EVAL_DIR

# evaluate each            
python run_evaluation.py --eval_jslf $BASELINE_RESULT_DIR/chatgpt0613long_model_selection3_ocw.jsonl  --eval_type ocw > $EVAL_DIR/baseline.out
python run_evaluation.py --eval_jslf $RIMS_RESULT_DIR/chatgpt0613long_rims_ocw.jsonl  --eval_type ocw > $EVAL_DIR/rims.out
python run_evaluation.py --eval_jslf $ABL_RESULT_DIR/ablation/chatgpt0613long_rims_ocw.jsonl  --eval_type ocw > $EVAL_DIR/rims_abl.out
