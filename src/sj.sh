# 1/ Configure vllm/openai server first and then do the followings
# 2/ edit placeholders and run the script


## constants
### n_jobs: for multiprocessing
NJOBS= # network bound job: larger better for faster results

### openllms names...
you need to edit endpoint at src/utils/llm_query_utils.py:L26 to point your openllm. followings now does not help but affects result directory namings.
LLAMA3_8B_IT=
PHI3_IT_small=
PHI3_IT_medium=
# ...


### rims prompts
GSM_RIMS_RW=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_newer_best_p2c2cot.pal2p2c.pal2cot.txt
GSM_RIMS_RW_1=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_cot2p2c.pal2cot.pal2p2c.txt
GSM_RIMS_RW_2=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt

OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
OCW_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt

NEWMATH_RIMS1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/newprompts/newmath_rims_1.txt
NEWMATH_RIMS2=prompt_construction_src/newer_prompts_3/math_ocw_prompts/newprompts/newmath_rims_2.txt
NEWMATH_RIMS3=prompt_construction_src/newer_prompts_3/math_ocw_prompts/newprompts/newmath_rims_3.txt

# 1-1: run baseline first
# T=0, n=1, greedy decoding experiments (temperature hardcoded inside run_inference.py for baseline run)
for MODEL in $LLAMA3_8B_IT $PHI3_IT_medium $PHI3_IT_small; do
    python run_inference.py baseline_inference \
        --backbone $MODEL \
        --gsm_jslf ../dataset/gsm8K_test.jsonl \
        --dataset_type gsm \
        --n_jobs $NJOBS

    python run_inference.py baseline_inference \
        --backbone $MODEL \
        --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
        --dataset_type ocw \
        --n_jobs $NJOBS

    python run_inference.py baseline_inference \
        --backbone $MODEL \
        --gsm_jslf ../dataset/MATH/MATH-full.jsonl \
        --dataset_type math \
        --n_jobs $NJOBS

done


# 1-2: rims run
BASELINE_RESULT_JSL_GSM=outputs/-------/n1_baseline.jsonl # path that prints out at the start/end of the above runs
BASELINE_RESULT_JSL_OCW=outputs/-----/n1_baseline.jsonl
BASELINE_RESULT_JSL_MATH=outputs/----/n1_baseline.jsonl
### gsm
for MODEL in $LLAMA3_8B_IT $PHI3_IT_medium $PHI3_IT_small; do
    for GSMPROMPT in $GSM_RIMS_RW $GSM_RIMS_RW_1 $GSM_RIMS_RW_2; do
        python run_inference.py rims_inference \
            --backbone $MODEL \
            --gsm_jslf $BASELINE_RESULT_JSL_GSM \
            --dataset_type gsm \
            --temperature 0. \
            --prompt_f $GSMPROMPT \
            --n 1 --n_jobs $NJOBS
    done

    for OCWPROMPT in $OCW_RIMS $OCW_RIMS_1; do
        python run_inference.py rims_inference \
            --backbone $MODEL \
            --gsm_jslf $BASELINE_RESULT_JSL_OCW \
            --dataset_type ocw \
            --temperature 0. \
            --prompt_f $OCWPROMPT \
            --n 1 --n_jobs $NJOBS
    done

    for MATHPROMPT in $NEWMATH_RIMS1 $NEWMATH_RIMS2 $NEWMATH_RIMS3; do
        python run_inference.py rims_inference \
            --backbone $MODEL \
            --gsm_jslf $BASELINE_RESULT_JSL_MATH \
            --dataset_type math \
            --temperature 0. \
            --prompt_f $MATHPROMPT \
            --n 1 --n_jobs $NJOBS
    done

done


# 2: self-consistency experiments
#   for n>1, temperatures are set to: rimsT = .7, cotT=.5, palT=.8

# 2-1: baseline + SC
for MODEL in $LLAMA3_8B_IT $PHI3_IT_medium $PHI3_IT_small; do
    python run_inference.py baseline_inference \
        --backbone $MODEL \
        --gsm_jslf ../dataset/gsm8K_test.jsonl \
        --dataset_type gsm \
        --n 15 \
        --n_jobs $NJOBS
done


# 2-2: rims run + SC
BASELINE_RESULT_JSL=outputs/somepath/thatprints/afterbaselinerun/or/see/code/n15_baseline.jsonl

for MODEL in $LLAMA3_8B_IT $PHI3_IT_medium $PHI3_IT_small; do
    for PROMPT in $GSM_RIMS $GSM_RIMS1 $GSM_RIMS2; do
        for T in 0.2 0.5; do
            python run_inference.py rims_inference \
                --backbone $MODEL \
                --gsm_jslf $BASELINE_RESULT_JSL \
                --dataset_type gsm \
                --temperature $T \
                --prompt_f prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt \
                --n 15 \
            --n_jobs $NJOBS
        done
    done
done


# 2-3: make SC 5, 10 from SC 15
RIMS_RESULT_JSL=outputs/-------/n15_rims_T0.2.jsonl

# configure `src/run_modif_SC_results.yaml` according to the guide inside

# and then run below

python run_modif_SC_results.py



# 3: evaluate

# n=1 results
for data in gsm ocw MATH; do
    python run_evaluation_new_n.py --ptn "outputs/*${data}*/**/n1_*.jsonl" --eval_type [gsm, ocw, math: depending on data] --outf [where to log .txt]
done
# SC results
for n in 5 10 15; do
    for data in gsm ocw MATH; do
        python run_evaluation_new_n.py --ptn "outputs/*${data}*/**/n${n}_*.jsonl" --eval_type [gsm, ocw, math: depending on data] --outf [where to log .txt]
    done
done
