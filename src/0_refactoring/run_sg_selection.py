"""
running
simple-greedy selection
(chul feels comfortable with calling the following as as above: Automatic Model Selection Reasoning which is our baseline https://arxiv.org/abs/2305.14333)

** note: selection algorithm supposed to run only where majority vote failed to reach the consensus **

for a designated dataset


- [ ] no self-consistency (n=1)
- [ ] self-consistency (n>1)

**
according to the author's code, self consistency is implemented with n-iterative call (https://github.com/XuZhao0/Model-Selection-Reasoning/blob/8ee494276e958c3f88332d4be64f8a746395f11c/src/selection_math.py#L301),
where we replace this with client.chat.completions.create(n=n)



----
resources:
    src/utils/model_selection_prompts.yaml # simple-greedy prompts for MATH, ocw dataset (gsm in math_prompt.py)
    src/utils/math_prompts.py


"""
import asyncio
from pathlib import Path
from typing import Dict, List, Union

import jsonlines as jsl
import pandas as pd
from query import ModelSelectionQuery


def question_from_indiv_row(row: Dict = None) -> str:
    return row["CoTQueryObject"]["query_message"][-1]["content"]


def solutions_from_indiv_row(
    row: Dict = None,
) -> Dict[List[str]]:
    indiv_dict = dict.fromkeys("cot pal p2c".split())
    indiv_dict["cot"] = row["CotQueryObject"]["contents"]
    indiv_dict["pal"] = row["PALQueryObject"]["contents"]
    indiv_dict["p2c"] = row["P2CQueryObject"]["contents"]
    return indiv_dict


def ds_type_from_indiv_row(row: Dict = None) -> str:
    return row["CoTQueryObject"]["meta"]["dataset_type"]


def unwrap_and_listify(d: Dict[str, List[str]]) -> Dict[str, str]:
    df = pd.DataFrame(d).to_dict(orient="records")
    return df.to_dict(orient="records")


async def simple_greedy_query(
    # indiv_records_path:Union[str, Path]="some.jsonl",
    row: Dict = None,
    temperature: float = 0.0,
    n: int = 1,
    seed: int = 777,
    backbone: str = "chatgpt1106",
):
    """
    process row and query selection (simple-greedy)
    """
    # read
    # indiv_records = list(jsl.open(indiv_records_path))
    dataset_type = ds_type_from_indiv_row(row=row)

    # init
    query_obj = ModelSelectionQuery(
        dataset_type=dataset_type,
    )

    # placeholders
    return_data = {}
    jobs = []

    question = question_from_indiv_row(row=row)
    cot_pal_p2c_sln_d = solutions_from_indiv_row(row=row)
    cot_pal_p2c_sln_d_1 = unwrap_and_listify(cot_pal_p2c_sln_d)

    max_tokens = 512

    for c_p_p2_sln in cot_pal_p2c_sln_d_1:
        query_params = {
            "question": question,
            "cot_pal_p2c_sln_d": c_p_p2_sln,  # specially for simple-greedy (prepare prompt)
            "temperature": temperature,
            "backbone": backbone,
            "n": n,
            "seed": seed,
            "max_tokens": max_tokens
            # "stop": # if needed stop let's add it to `model_selection_prompts.yaml` and load.
        }

        jobs.append(query_obj.async_query(**query_params))

    for contents, query_message, resp, meta in await asyncio.gather(*jobs):
        if not return_data:
            return_data[meta["method_obj"]] = {
                "contents": [contents],
                "query_message": [query_message],
                "resp": [resp],
            }
        else:
            return_data[meta["method_obj"]]["contents"].append(contents)
            return_data[meta["method_obj"]]["query_message"].append(query_message)
            return_data[meta["method_obj"]]["resp"].append(resp)
    return return_data
