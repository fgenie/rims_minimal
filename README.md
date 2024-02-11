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
                --outdir dbgoutdir/ \
                --gsm_jslf ../dataset/dbg_gsm.jsonl \ # must contain dataset_type in its name 
                --dataset_type gsm # [ocw, math, gsm, svamp] # this will affect majority voting
                [--start_idx 0 ] # can start running from the middle of the data
                [--dbg] # runs with tqdm instead of pqdm
```


### 2 rims algorithm run the 1's result (it will skip the non-conflict examples!)
```bash
python run_inference.py  rims_inference  \
                --backbone  chatgpt0613long  \
                --outdir dbgoutdir/ \
                --prompt_f   prompt_construction_src/prep_rims_prompts/gsm_prompts/3_reflectonce_p2c2cot.pal2p2c.pal2cot.txt_rm_ans   \ # current best prompt
                --gsm_jslf dbgoutdir/chatgpt0613long_model_selection3_gsm.jsonl \
                --dataset_type gsm # [ocw, math, gsm, svamp]
```

### 3 evaluate
```bash
# baseline/rims result
python run_evaluation.py --eval_jslf dbgoutdir/chatgpt0613long_model_selection3_gsm.jsonl --eval_type [gsm|svamp|ocw|math]
# individual method (i.e. cot, pal, p2c) results
python run_evaluation_each.py --eval_jslf dbgoutdir/chatgpt0613long_model_selection3_gsm.jsonl --eval_type [gsm|svamp|ocw|math]
# will write in result.txt 
```

## last tweaks
 - [ ] math-full categorical / level analyses (what method preferred)
 - [x] ocw numeric / symbolic analyses
 - [ ] self-consistency condition of baseline, T>0 experiment

## todo candids
 - prompts with symbolic examples (math ocw)
 - openLLM experiments
 - analyses on the results so far
