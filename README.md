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
- [x] implement n>1 case
  - [x] baseline_complete_row
  - [x] indiv_inference
  - [x] rims_complete_row
    - [ ] 왜 raw output만 output파일에 안남는 것일까??
  - [x] dbg
  - [ ] `run_SC_truncation.py`: will reduce SC\<15 results from SC=15 jsonlines file
  - [ ] `run_evaluation_new_n.py`: will eval the results
- [ ] ?conversion to async for performance? -- required, but not for chatgpt
  - ~4x speedup expected... for SC=15
  - chatgpt + SC 15 + math (5000 rows) --> 36 hours
    - gpt4turbo? :dead:

### SJ
- [ ] opensource LLM
  - vllm/openai 서버를 태형님 DGX에서 구동한다
  - 코드 일부 수정
    - `llm_query_utils.py` 에서 client 부분을 uncomment하고 `base_url` 설정. AzureClient는 comment처리
    - `client.chat.completion.create()` 의 인자 중에서 vllm_endpoint가 지원하지 않는 kwarg를 배제한다 (e.g. seed? model?)
  - `sj.sh` 의 빈 칸들을 채워서 구동한다
    - baseline model selection의 경로 등
  - 이렇게 했을 때 혹시 간단하게 해결할 수 없어보이는 문제 있으면 문의
