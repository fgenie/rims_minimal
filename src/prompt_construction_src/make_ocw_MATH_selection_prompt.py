from utils.llm_query_utils import query_cot, query_pal, query_plancode, \
                                query_selection, get_select_prompt, \
                                safe_execute_turbo, extract_num_turbo
from utils.math_util import is_equiv, \
                            is_equiv_ocw, \
                            normalize_final_answer         

from fire import Fire 
from omegaconf import OmegaConf 

from functools import partial
from typing import Callable, Literal, Any
from collections import defaultdict


def gsm_check_answer(a1, a2):
    try: 
        decision = abs(a1-a2)<1e-3
    except Exception as e:
        print(e)
        decision = None
    return decision

def ocw_check_answer(a1, a2):
    """
    check if a1 and a2 are equivalent in ocw
    """
    decision = is_equiv_ocw(normalize_final_answer(a1), normalize_final_answer(a2))
    return decision

def math_check_answer(a1, a2):
    """
    check if a1 and a2 are equivalent in math
    """
    decision = is_equiv(normalize_final_answer(a1), normalize_final_answer(a2))
    return decision

def repeat_query_until(question:str="", 
                       method:Literal["cot", "pal", "p2c"]="",
                       dataset_type:Literal["ocw", "math", "gsm"]="",
                       inference_kwargs:dict=None, 
                       answer:Any=None,
                       n:int=10):
    """
    returns
    llm_responses: dict =
        dict(
            correct:list = slns,
            incorrect:list = slns 
        )
    """
    assert question
    assert method in ["cot", "pal", "p2c"]
    assert dataset_type in ["ocw", "math", "gsm"]
    assert inference_kwargs
    assert answer

    # method determines query & exec_function
    if method == "cot":
        query_func = query_cot
        exec_func = extract_num_turbo
    elif method == "pal":
        query_func = query_pal
        exec_func = safe_execute_turbo
    elif method == "p2c":
        query_func = query_plancode
        exec_func = safe_execute_turbo
    else:
        raise ValueError(f"method: {method} is not supported")

    # dataset_type determines evaluation function
    if dataset_type == "gsm":
        check_answer = gsm_check_answer
    elif dataset_type == "ocw":
        check_answer = ocw_check_answer
    elif dataset_type == "math":
        check_answer = math_check_answer
    else:
        raise ValueError(f"dataset_type: {dataset_type} is not supported")
            
    inference_kwargs.update({"n":n})

    sln_lst, *_ = query_func(question, **inference_kwargs) # cot|pal : sln, msgs / p2c : sln, plan, msgs
    
    res = defaultdict(list)
    for sln in sln_lst:
        pred = exec_func(sln)
        key = "correct" if check_answer(pred, answer) else "incorrect"
        res[key].append(sln)
    
    return res
        
    

def main():
    # 다음 둘에 fewshot에 활용된 question들이 달려있다.
    # ocw_MATH_prompts.math_cot.user 
    # ocw_MATH_prompts.ocw_cot.user
    ymlf = "../utils/ocw_MATH_prompts.yaml"
    prompt_d = OmegaConf.load(ymlf)

    math_questions = prompt_d.math_cot.user
    math_answers = prompt_d,math_cot.answers

    ocw_questions = prompt_d.ocw_cot.user
    ocw_answers = prompt_d.ocw_cot.answers

    # inference_kwargs
    cot_kwargs = dict(
        # question: str, 
        dataset_type = "tobefilled", 
        temperature = 0.5, 
        backbone = "GPT35", 
        seed=777,
    )
    pal_kwargs = dict(
        temperature=0.7, 
        backbone='GPT35', 
        seed=777,
    )
    p2c_kwargs = dict(
        # question: str,  
        # n=1,
        plan_temperature = 0.5,
        code_temperature = 0.7,
        backbone = "GPT35",
        seed = 777,
    )

    dataset_types = ["ocw", "math"]
    methods = ["cot", "pal", "p2c"]
    

if __name__ == "__main__":
    Fire(main)
    
