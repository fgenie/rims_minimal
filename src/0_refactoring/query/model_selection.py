"""
code for model selection (i.e. simple-greedy)
"""
from pathlib import Path
from typing import Any, Dict, List, Literal

from base_query import BaseQueryObject


class ModelSelectionQuery(BaseQueryObject):
    def __init__(self, dataset_type):
        self.dataset_type = dataset_type

    async def prepare_query(
        self,
        question: str,
        cot_pal_p2c_sln_d: Dict,
        **kwargs,  # to avoid overiding base class
    ):
        return get_select_prompt2(
            question,
            cot_pal_p2c_sln_d=cot_pal_p2c_sln_d,
            dataset_type=self.dataset_type,
        )

    def query_error_msg(self, query_message):
        return False, None


### getting prompts for each method ###
def get_select_prompt2(
    question: str,
    cot_pal_p2c_sln_d: dict = None,
    dataset_type: Literal["gsm", "svamp", "ocw", "math"] = None,
) -> List[Dict[str, str]]:
    # open up prompt template yaml file
    THIS_PARENT = Path(__file__).parent.resolve()
    prompt_yml = THIS_PARENT / "model_selection_prompts.yaml"
    prompt_d: Dict[str, Any] = yaml.full_load(open(prompt_yml))

    select_prompt_key = f"{dataset_type}_select"
    ds_prom_d: Dict[str, Any] = prompt_d[select_prompt_key]

    # process with the question/solutions of interest to result in gpt_messages
    system: str = ds_prom_d["system"]
    fewshots_user = ds_prom_d["user"]
    fewshots_assistant = ds_prom_d["assistant"]

    # make user's query from (question, cot_pal_p2c_sln_d)
    q = question  # data['question']
    to_replace_keys = "{COT_SOLUTION} {PAL_SOLUTION} {P2C_SOLUTION}"
    user_tmp: str = ds_prom_d["user_tmp"]

    # 1. fill the quesiton
    user_tmp = user_tmp.replace("{QUESTION}", q)
    # 2. fill the solutions

    for to_replace, to_be in zip(to_replace_keys.split(), cot_pal_p2c_sln_d.values()):
        user_tmp = user_tmp.replace(to_replace, to_be)
    user_attempt = user_tmp

    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": system},
    ]
    for fu, fa in zip(fewshots_user, fewshots_assistant):
        usr = {"role": "user", "content": fu}
        astnt = {"role": "assistant", "content": fa}
        msgs.append(usr)
        msgs.append(astnt)

    msgs.append({"role": "user", "content": user_attempt})

    return msgs
