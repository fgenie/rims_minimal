

# be sure to run baseline(simple-greedy model selection) and then run rims on results of baseline

## constants
### n_jobs: for multiprocessing
NJOBS=

### openllms
OPEN=
LLM=
YOU=
WANT=

### rims prompts
GSM_RIMS_RW=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_newer_best_p2c2cot.pal2p2c.pal2cot.txt
GSM_RIMS_RW_1=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_cot2p2c.pal2cot.pal2p2c.txt
GSM_RIMS_RW_2=prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt

OCW_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.pal-cot__.txt
OCW_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_ocw_p2c-cot.pal-p2c.cot-p2c__.txt

MATH_RIMS=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt
MATH_RIMS_1=prompt_construction_src/newer_prompts_3/math_ocw_prompts/rims_math_p2c-cot.pal-p2c.pal-cot__.txt1

# T=0, n=1, greedy decoding experiments
## baseline run
for MODEL in $OPEN $LLM $YOU $WANT; do
    python run_inference.py  baseline_inference \
                    --backbone $MODEL \
                    --gsm_jslf ../dataset/gsm8K_test.jsonl \
                    --dataset_type gsm

    python run_inference.py baseline_inference \
                    --backbone $MODEL \
                    --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
                    --dataset_type ocw

    python run_inference.py baseline_inference \
                    --backbone $MODEL \
                    --gsm_jslf ../dataset/MATH/MATH-full.jsonl \
                    --dataset_type math
done


## rims run
BASELINE_RESULT_JSL_GSM=
BASELINE_RESULT_JSL_OCW=
BASELINE_RESULT_JSL_MATH=
### gsm
for MODEL in $OPEN $LLM $YOU $WANT; do
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

    for MATHPROMPT in $MATH_RIMS $MATH_RIMS_1; do
        python run_inference.py rims_inference \
            --backbone $MODEL \
            --gsm_jslf $BASELINE_RESULT_JSL_MATH \
            --dataset_type math \
            --temperature 0. \
            --prompt_f $MATHPROMPT \
            --n 1 --n_jobs $NJOBS
    done

done


# self-consistency experiments
#   for n>1, temperatures are set to: rimsT = .7, cotT=.5, palT=.8
#   run for (SC) n=15, later will reduce SC<15 results from outputs

##example scripts here for gsm.
##do the same for the others as well above.

for MODEL in $OPEN $LLM $YOU $WANT; do
    python run_inference.py baseline_inference \
        --backbone $MODEL \
        --gsm_jslf ../dataset/gsm8K_test.jsonl \
        --dataset_type gsm \
        --n 15 \
        --n_jobs $NJOBS
done


PREV_RESULT_JSL=

for MODEL in $OPEN $LLM $YOU $WANT; do
    python run_inference.py rims_inference \
        --backbone $MODEL \
        --gsm_jslf $PREV_RESULT_JSL \
        --dataset_type gsm \
        --temperature 0.7 \
        --prompt_f prompt_construction_src/newer_prompts_3/renewed_gsm_prompts/rewrote.p2c_gsm_pal2p2c.cot2p2c.cot2pal.txt \
        --n 15 \
        --n_jobs $NJOBS
done
