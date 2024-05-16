# 1/ Configure vllm/openai server first and then do the followings
# 2/ edit placeholders and run the script

# set up
# python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --dtype bfloat16 --tensor-parallel-size 2 --served-model-name Meta-Llama-3-8B-Instruct --disable-log-stats
# python -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-Instruct-v0.2 --dtype bfloat16 --tensor-parallel-size 2 --served-model-name Mistral-7B-Instruct-v0.2 --disable-log-stats --chat-template mistral_chat_template.txt
# python -m vllm.entrypoints.openai.api_server --model google/gemma-1.1-2b-it --dtype bfloat16 --tensor-parallel-size 2 --served-model-name gemma-1.1-2b-it --disable-log-stats --chat-template gemma_chat_template.txt


## constants
### n_jobs: for multiprocessing
NJOBS=$1 # network bound job: larger better for faster results

### openllms names...
# you need to edit endpoint at src/utils/llm_query_utils.py:L26 to point your openllm. followings now does not help but affects result directory namings.
# MODEL_NAME=Meta-Llama-3-8B-Instruct 
# MODEL_NAME=Mistral-7B-Instruct-v0.2
MODEL_NAME=Phi-3-mini-128k-instruct #gemma-1.1-2b-it
# ...


### rims prompts
GSM_RIMS_RW=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_newer_best_p2c2cot.pal2p2c.pal2cot.txt
GSM_RIMS_RW_1=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_cot2p2c.pal2cot.pal2p2c.txt
GSM_RIMS_RW_2=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt

OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
OCW_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt

MATH_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt
MATH_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt1

# 1-1: run baseline first
# T=0, n=1, greedy decoding experiments (temperature hardcoded inside run_inference.py for baseline run)
python run_inference.py baseline_inference \
    --backbone $MODEL_NAME \
    --gsm_jslf ../dataset/gsm8K_test.jsonl \
    --dataset_type gsm \
    --n_jobs $NJOBS

python run_inference.py baseline_inference \
    --backbone $MODEL_NAME \
    --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
    --dataset_type ocw \
    --n_jobs $NJOBS

# python run_inference.py baseline_inference \
#     --backbone $MODEL_NAME \
#     --gsm_jslf ../dataset/MATH/MATH-full.jsonl \
#     --dataset_type math \
#     --n_jobs $NJOBS


# 1-2: rims run
# path that prints out at the start/end of the above runs
BASELINE_RESULT_JSL_GSM=outputs/gsm8K_test_dt.gsm/$MODEL_NAME/model_selection_prompts/n1_baseline_T0.5_0.8_last.jsonl
BASELINE_RESULT_JSL_OCW=outputs/ocw_course_dt.ocw/$MODEL_NAME/model_selection_prompts/n1_baseline_T0.5_0.8_last.jsonl
### gsm
for GSMPROMPT in $GSM_RIMS_RW $GSM_RIMS_RW_1 $GSM_RIMS_RW_2; do
    python run_inference.py rims_inference \
        --backbone $MODEL_NAME \
        --gsm_jslf $BASELINE_RESULT_JSL_GSM \
        --dataset_type gsm \
        --temperature 0. \
        --prompt_f $GSMPROMPT \
        --n 1 \
        --n_jobs $NJOBS
done

### ocw
for OCWPROMPT in $OCW_RIMS $OCW_RIMS_1; do
    python run_inference.py rims_inference \
        --backbone $MODEL_NAME \
        --gsm_jslf $BASELINE_RESULT_JSL_OCW \
        --dataset_type ocw \
        --temperature 0. \
        --prompt_f $OCWPROMPT \
        --n 1 \
        --n_jobs $NJOBS
done


# # 2: self-consistency experiments
# #   for n>1, temperatures are set to: rimsT = .7, cotT=.5, palT=.8

# 2-1: baseline + SC
# python run_inference.py baseline_inference \
#     --backbone $MODEL_NAME \
#     --gsm_jslf ../dataset/gsm8K_test.jsonl \
#     --dataset_type gsm \
#     --n 15 \
#     --n_jobs $NJOBS

# python run_inference.py baseline_inference \
#     --backbone $MODEL_NAME \
#     --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
#     --dataset_type ocw \
#     --n 15 \
#     --n_jobs $NJOBS


# 2-2: rims run + SC
BASELINE_RESULT_GSM_JSL=outputs/gsm8K_test_dt.gsm/Meta-Llama-3-8B-Instruct/model_selection_prompts/n15_baseline.jsonl
BASELINE_RESULT_OCW_JSL=outputs/ocw_course_dt.ocw/Meta-Llama-3-8B-Instruct/model_selection_prompts/n15_baseline.jsonl

# for T in 0.2 0.5; do
#     python run_inference.py rims_inference \
#         --backbone $MODEL_NAME \
#         --gsm_jslf $BASELINE_RESULT_GSM_JSL \
#         --dataset_type gsm \
#         --temperature $T \
#         --prompt_f prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt \
#         --n 15 \
#         --n_jobs $NJOBS
# done

# for T in 0.2 0.5; do
#     python run_inference.py rims_inference \
#         --backbone $MODEL_NAME \
#         --gsm_jslf $BASELINE_RESULT_OCW_JSL \
#         --dataset_type ocw \
#         --temperature $T \
#         --prompt_f prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt \
#         --n 15 \
#     --n_jobs $NJOBS
# done

# for T in 0.2 0.5; do
#     python run_inference.py rims_inference \
#         --backbone $MODEL_NAME \
#         --gsm_jslf $BASELINE_RESULT_OCW_JSL \
#         --dataset_type ocw \
#         --temperature $T \
#         --prompt_f prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt \
#         --n 15 \
#     --n_jobs $NJOBS
# done


# # 2-3: make SC 5, 10 from SC 15
# RIMS_RESULT_JSL=outputs/-------/n15_rims_T0.2.jsonl

# # configure `src/run_modif_SC_results.yaml` according to the guide inside

# # and then run below

# python run_modif_SC_results.py



# # 3: evaluate

# # n=1 results
for data in gsm ocw; do
    python run_evaluation_new.py --ptn outputs/*${data}*/**/n1_*.jsonl --eval_type $data --outf testout.txt
done

for data in gsm ocw; do
    python run_evaluation_new.py --ptn outputs/*${data}*/**/rims_*.jsonl --eval_type $data --outf testout.txt
done
# # SC results
# for n in 15; do
#     for data in gsm ocw; do
#         python run_evaluation_new_n.py --ptn outputs/*${data}*/**/n${n}_*.jsonl --eval_type $data --outf testout.out
#     done
# done
