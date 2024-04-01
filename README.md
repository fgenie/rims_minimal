# RIMS (<u>R</u>eflective H<u>i</u>nt <u>M</u>odel <u>S</u>election)

## How to Contribute
```
pip install pre-commit
pre-commit install
```

## experiment scripts
- `0_run_baseline_mar23.sh`
- `1_run_rims_mar23.sh`

### commands in brief
```bash
# first, run simple-greeedy
python run_inference.py  baseline_inference \
                --backbone chatgpt1106 \
                --gsm_jslf ../dataset/ocw/ocw_course.jsonl \
                --dataset_type ocw

# 2nd, run rims on the result of simple-greedy
python run_inference.py rims_inference \
    --backbone chatgpt1106 \
    --gsm_jslf ${MATH_INFERRED} \
    --dataset_type math \
    --prompt_f ${PROMPT}

# 3rd, evaluate the target directories with wildcard expression
python run_evaluation_new.py --ptn "outputs/MATH-full_dt.math/chatgpt1106/*/*jsonl" --eval_type math --outf math1106_results.txt
```

## reset experiment prompts
- [ ] experiments further
     - [ ] self-consistency condition of baseline, T>0 experiment
     - [ ] opensource llm (deepseek math, llama)
     - [ ] gpt4
