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
- [x] tests
    - azure, evaluation, parsing, fewshot harvesting, how these affects the older results...
- [x] CoT parsing for OCW, MATH: isn't `extract_num_turbo()` too specific for GSM and SVAMP?
    - [x] indeed! --> implemented `extract_ans_from_cot_MATHnOCW`   
- [x] problem of latex/sympy evaluation
    - [x] code execution: `try` `sp.latex(solution())` at the end (this do not affect gsm)
    - [x] bunch of evaluation fixes and tests
- [x] harvest wrong / correct sets and prepare the followings
    - [x] if not applicable, create example with claude sonnet. 
    - [ ] fewshots_p2c_math_ocw.txt (WIP)
- [x] p2c prompts: coding challenges
    - [x] MBPP prompts in the paper
- [x] cot prompts: dataset-specific
    - [x] OCW 
    - [x] MATH
- [x] (NEW!) pal prompts: dataset-specific
    - [x] OCW
    - [x] MATH 
- [ ] selection prompts: dataset-specific
    - [x] GSM, util
    - [ ] OCW (WIP)
    - [ ] MATH
    - [ ] renew `get_prompt()` 
- [ ] RIMS prompts: dataset-specific
    - [ ] OCW
    - [ ] MATH
- [ ] apply `@utils.cost_tracking.CountTokens`
    - [x] 1 more output for `token_info` dict
        - query_f's : _query, query_cot, query_selection, query_rims_inference 
    - [x] CountTokens need to crunch the `token_info`
- [ ] dbg (`run_inference.py`)
    - [ ] python run_inference.py baseline_inference
    - [ ] python run_inference.py rims_inference 
        