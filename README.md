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

## TODO
### SI
- [ ] `openlimit-->ratelimit`
- [x] why several (~8) lines missing from OCW_RIMS@SC5 results ? --> api error of baseline + failure in rims
  - [x] first baseline fails (api quota) --> removed in rims + failures
- [x] `run_SC_truncation.py`: will reduce SC\<15 results from SC=15 jsonlines file
  - [x] merging
  - [x] splitting
  - [x] dbg
  - [ ] yaml file and run
- 2_1_leftovers.sh
  - [x] chatgpt -math SC5
  - [ ] gpt4turbo -math SC5 ~4k rows
- 2_2_leftovers_rims_after_merge.sh
  - [ ] chatgpt1106
  - [ ] math gpt4turbo
  - [x] others gpt4turbo
- [x] do leftovers
  - error lines of baseline run
  - missing lines of rims run
- [ ] analyses
  - [ ] SC+single method accuracy
  - [ ] did_reflect distribution in SC15
  - [ ] detailed analyses on what makes SC so helpful (expected domination of majvote but... what proportion did rims/simple did the job?)
    - [ ] how many examples had highly un-agreed answers



### 전달시
openllm-sj 브랜치에서
- tunneling 으로 dgx api query 가능한지 확인
- sj.sh 작동하는지 확인, 숫자도 얼핏
- 함수 분야별로 쪼개기, 새로 짤 부분 만들기...
