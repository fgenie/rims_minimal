# rims_minimal
이전 코드베이스가 엉망인 관계로 제공함 <br>
시작점: https://github.com/fgenie/Model-Selection-Reasoning/tree/59debd8441e7cb4b7d733f256b20870073e36c08


## What to do before run
place `openai_key.txt` into `utils/`

## How to Contribute

```
pip install pre-commit
pre-commit install
```

## How to run
see `greedy_experiment_example_script.sh` it provides one-pot experiment example.
### 1 model-selection-resoning baseline
```bash
python run_inference.py  baseline_inference \
                --backbone chatgpt0613long \
                --outdir $BASELINE_RESULT_DIR \
                --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
                --dataset_type ocw

```


### 2 rims algorithm run the 1's result (it will skip the non-conflict examples!)
```bash
python run_inference.py rims_inference \
                            --prompt_f $V3PROMPT \
                            --gsm_jslf $BASELINE_RESULT_DIR/chatgpt0613long_model_selection3_ocw.jsonl \
                            --dataset_type ocw \
                            --backbone chatgpt0613long \
                            --outdir $RIMS_RESULT_DIR 
```

### 3 evaluate
```bash
mkdir -p $EVAL_DIR

# evaluate each            
python run_evaluation.py --eval_jslf $BASELINE_RESULT_DIR/chatgpt0613long_model_selection3_ocw.jsonl  --eval_type ocw > $EVAL_DIR/baseline.out
python run_evaluation.py --eval_jslf $RIMS_RESULT_DIR/chatgpt0613long_rims_ocw.jsonl  --eval_type ocw > $EVAL_DIR/rims.out
python run_evaluation.py --eval_jslf $ABL_RESULT_DIR/ablation/chatgpt0613long_rims_ocw.jsonl  --eval_type ocw > $EVAL_DIR/rims_abl.out
```

## to check
 - [ ] self-consistency condition of baseline, T>0 experiment

## reset experiment prompts 
 - [x] `query_rims_inference()` do not require max_token == 2048, long p2c to pal reflection blurb is around 700, so I set its value to 1024, which would be 1.5x of the long blurb (observation: max reflection = 2 times)
 - [ ] prompts
    - [x] p2c prompts
        - [x] MBPP prompts in the paper
    - [x] cot prompts
        - [x] OCW
        - [x] MATH
    - [x] util test for aboves 
    - [x] does OCW parsing function, changed, works better than before??
        - [x] changing ocw parsing function does not do any good...
        - updating eval makes sense
    - [x] isn't `num_extract_turbo()` too specific for GSM and SVAMP?
        - [x] indeed! and found it does not parse scientific-formatting as well so I've fixed it.   
        - [x] re-evaluate the previous `MATH`, `ocw_courses` results 
    - [x] harvest wrong / correct sets and prepare the followings
        - done

    - [ ] selection prompts
        - [x] GSM, util
        - [ ] OCW
        - [ ] MATH
        - [ ] double check if the cot path of the solution is also appropriate (not correct by chance)
    - [ ] RIMS prompts
        - [ ] OCW
        - [ ] MATH
        - [ ] util test 
 - [ ] `OPENAI` client to `AzureOPENAI`
    - [x] endpoint and key, client setting 
    - [x] modelname --> deployment name
        - see `rims_minimal/src/prompt_construction_src/tests/test_azure.py`
    - [x] API version test
