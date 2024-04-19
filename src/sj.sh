

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
BASELINE_RESULT_JSL_GSM=somepath/thatprints/afterbaselinerun/or/see/code/baseline.jsonl
BASELINE_RESULT_JSL_OCW=somepath/thatprints/afterbaselinerun/or/see/code/baseline.jsonl
BASELINE_RESULT_JSL_MATH=somepath/thatprints/afterbaselinerun/or/see/code/baseline.jsonl
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


PREV_RESULT_JSL=somepath/thatprints/afterbaselinerun/or/see/code/n15_baseline.jsonl

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




# gpt-3.5-turbo-1106 output token stats
# {'GSM': {'cot': {'50%': 148.0,
#                  '95%': 277.0,
#                  '99%': 354.73000000000025,
#          'p2c': {'50%': 274.5,
#                  '95%': 401.0,
#                  '99%': 458.73000000000025,
#                  'max': 526.0,
#          'pal': {'50%': 132.0,
#                  '95%': 224.54999999999995,
#                  '99%': 277.5500000000004,
#                  'max': 337.0,
#          'plan': {'50%': 54.0,
#                   '95%': 92.0,
#                   '99%': 110.0,
#                   'max': 142.0,
#          'rims': {'50%': 459.5,
#                   '95%': 855.6499999999999,
#                   '99%': 1321.0899999999979,
#  'GSM_SELECTION': {'cot': {'50%': 148.0,
#                            '95%': 276.70000000000005,
#                            '99%': 354.82000000000016,
#                            'max': 1024.0,
#                    'p2c': {'50%': 275.0,
#                            '95%': 400.0,
#                            '99%': 458.82000000000016,
#                            'max': 526.0,
#                    'pal': {'50%': 131.0,
#                            '95%': 223.70000000000005,
#                            '99%': 277.7000000000003,
#                            'max': 337.0,
#                    'plan': {'50%': 54.0,
#                             '95%': 92.0,
#                             '99%': 110.0,
#                             'max': 142.0,
#                    'selection': {'50%': 38.0,
#                                  '95%': 51.849999999999994,
#                                  '99%': 59.0,
#                                  'max': 60.0,
#  'MATH': {'cot': {'50%': 198.0,
#                   '95%': 546.9000000000005,
#                   '99%': 931.2999999999984,
#                   'max': 1024.0,
#           'p2c': {'50%': 258.0,
#                   '95%': 498.0,
#                   '99%': 661.2999999999984,
#                   'max': 1024.0,
#           'pal': {'50%': 93.0,
#                   '95%': 250.0,
#                   '99%': 343.8599999999997,
#                   'max': 618.0,
#           'plan': {'50%': 53.0,
#                    '95%': 105.0,
#                    '99%': 141.0,
#                    'max': 1024.0,
#           'rims': {'50%': 519.0,
#                    '95%': 980.7999999999993,
#                    '99%': 1383.24,
#                    'max': 1624.0,
#  'MATH_SELECTION': {'cot': {'50%': 204.0,
#                             '95%': 560.0,
#                             '99%': 925.96,
#                             'max': 1024.0,
#                     'p2c': {'50%': 272.0,
#                             '95%': 514.6000000000004,
#                             '99%': 676.96,
#                             'max': 1024.0,
#                     'pal': {'50%': 94.0,
#                             '95%': 251.80000000000018,
#                             '99%': 346.96000000000004,
#                             'max': 618.0,
#                     'plan': {'50%': 53.0,
#                              '95%': 105.80000000000018,
#                              '99%': 143.0,
#                              'max': 1024.0,
#                     'selection': {'50%': 47.0,
#                                   '95%': 80.25,
#                                   '99%': 102.90000000000009,
#                                   'max': 148.0,
#  'OCW': {'cot': {'50%': 326.0,
#                  '95%': 800.4,
#                  '99%': 1020.6399999999999,
#                  'max': 1024.0,
#          'p2c': {'50%': 416.0,
#                  '95%': 755.6999999999998,
#                  '99%': 981.5199999999999,
#                  'max': 1024.0,
#          'pal': {'50%': 129.0,
#                  '95%': 249.79999999999995,
#                  '99%': 281.52,
#                  'max': 358.0,
#          'plan': {'50%': 88.0,
#                   '95%': 223.09999999999985,
#                   '99%': 392.79999999999984,
#                   'max': 1024.0,
#          'rims': {'50%': 598.5,
#                   '95%': 1006.8499999999999,
#                   '99%': 1128.8999999999999,
#                   'max': 1399.0,
#  'OCW_SELECTION': {'cot': {'50%': 268.5,
#                            '95%': 601.7499999999999,
#                            '99%': 758.2499999999998,
#                            'max': 1024.0,
#                    'p2c': {'50%': 366.5,
#                            '95%': 561.8499999999999,
#                            '99%': 701.5999999999998,
#                            'max': 810.0,
#                    'pal': {'50%': 179.5,
#                            '95%': 404.54999999999995,
#                            '99%': 468.18999999999994,
#                            'max': 606.0,
#                    'plan': {'50%': 62.0,
#                             '95%': 133.54999999999998,
#                             '99%': 186.82,
#                             'max': 228.0,
#                    'selection': {'50%': 72.0,
#                                  '95%': 103.8,
#                                  '99%': 126.52000000000001,
#                                  'max': 139.0,
