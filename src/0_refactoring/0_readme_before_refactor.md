# Criterion for refactoring
* we aim to reproduce the similar or better `llama-3-8b-it` result (`llama-reference.report`)
  * so we start from the code where it was executed: branch name = `openllm-sj`


# Big picture
* multiprocessing --> asynchronous
* Integrated `querying` and `text processing` need to be in separate steps
  * `run_*.py` will result in raw query output records (**no format existant for now**)
  * `postprocess_rawouts.py` will process the results from above into some integrated file... (considering to follow the previous format, or **maybe more readable and conciser one**)
    * current formats are
* `*_functions.py` will provide funtionalities for above logics


# 리팩터링 전 가장 먼저 읽고 시작
그냥 뒤엎는거랑 다름이 없어보이긴 하지만... 그 편이 나을 지도 모릅니다. 최소한 가이드는 다 있는 상태.
완전히 역할분담이 되진 않을 것이지만 그 점이 교차검증에 도움이 되는 면도 있다고 생각합니다.
제가 생각하는 좋은 작업효율은 아마도
- querying 관련 구현을 맡아주시고
- 제가 text processing, 쪽을 맡는 것
아마 양방으로 도움얻을 일이 있을 것 같습니다

이렇게 한 이유는 데이터가 버전따라 키나 필드가 조금씩 달라지는 경우들도 있고 해서 가이드가 쉽지 않을 것 같다는 판단이 이런 결정을 이끌었습니다.

채점코드는 src/utils/math_util.py 에 나온대로 minerva (lm_eval_harness)에서 코드를 따왔고 latex 채점이 너무 안좋아서 중간에 고쳤습니다 (물론 이게 repo 목적에는 맞지 않습니다만...)
https://github.com/fgenie/rims_minimal/issues/37#issuecomment-1967256560


# Left for later
- batch API for openAI model experiments (some thangs left...) --> 손선일...!
- 수행할실험
  - chatgpt
    - T>0, sc 10
      - simple greedy (1) : math
      - rims (1): math
    - T=0, no SC (0 = all done)
  - gpt-4-1106-preview --> not sure will do
    - T=0 no SC (0 = all done)
    - T>0, sc5
      - simple greedy (1): math
      - rims (3): math, ocw, gsm
- `src/run_evaluation_new.py`
- `src/run_evaluation_new_n.py` --> will require some changes later
- `src/run_modif_SC_results.py` --> code for merging SC5 + SC10 results to make SC15 result... product of inperformant code.



# Issues with the current implementation
* most of the runtime errors are hidden by `try-except`, which is hard to debug
* parallel processing, even with `pqdm` package makes debugging real hard. `asyncio` will work our purpose


# Some explanations about original Implementation
* first: I'm sorry to let you read my code... seriously. I appreciate learning from you.
* some notions
  * `plan2code` is nickname for https://arxiv.org/abs/2303.06689
    * it is prompt chaining: question --> plan --> code
    * it has no open implementation, so I just reproduced their work. Exactly followed their prompt of MBPP fewshots which confirmed to work OK in our datasets.
    * in n>1 scenario, I compromised performance with correctness. it should have been like:
      * question -> {plan}_N -> {code}_N
      * question -> plan -> {code}_N (in my current code)
  * (n>1) SC was originally implemented as a loop of n=1 cycle. which means
    * for simple_greedy: indiv n=1 --> majority vote ([if fails] --> selection) (each selection will take their previous indiv solution as prompting resources)
    * for rims: likewise, rims will only run for failed-to-choose-majorities, but this could be implemented with `client.chat.completions.create(n=n)` as it does not actually takes the solution of the previous.





OPENAI_API_BASE=http://localhost:8000/v1 python run_baseline.py --gsm_jslf=some_dir/gsm8K_test.jsonl --dataset_type=gsm --backbone=Meta-Llama-3-8B-Instruct

OPENAI_API_BASE=http://localhost:8000/v1 python run_baseline.py --gsm_jslf=/data/recoteam_583/joel/opensource/rims_minimal/dataset/gsm8K_test.jsonl --dataset_type=gsm --backbone=Meta-Llama-3-8B-Instruct --retry_error_in_result_file_path=outputs/gsm8K_test_dt.gsm/Meta-Llama-3-8B-Instruct/n1_baseline_raw_query_result.jsonl
