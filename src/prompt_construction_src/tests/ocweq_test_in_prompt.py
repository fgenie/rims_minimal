from utils.llm_query_utils import query_cot, query_pal, query_plancode, \
                                query_selection, get_select_prompt, \
                                safe_execute_turbo, extract_num_turbo, \
                                extract_ans_from_cot_MATHnOCW
from utils.math_util import is_equiv, \
                            is_equiv_ocw, \
                            normalize_final_answer, normalize_symbolic_expression         

from fire import Fire 
# from omegaconf import OmegaConf 
import yaml
import json


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
    a1, a2 = map(str, [a1, a2])
    decision_new = is_equiv_ocw(a1, a2, use_sym_exp_normalizer=True) 
    decision_old = is_equiv_ocw(a1, a2, use_sym_exp_normalizer=False) 
    return decision_new, decision_old

def math_check_answer(a1, a2):
    """
    check if a1 and a2 are equivalent in math
    """
    a1, a2 = map(str, [a1, a2])
    decision = is_equiv(normalize_final_answer(a1), normalize_final_answer(a2))
    return decision

def correct_incorrect_query_results(question:str="", 
                       method:Literal["cot", "pal", "p2c"]="",
                       dataset_type:Literal["ocw", "math", "gsm"]="",
                       inference_kwargs:dict=None, 
                       answer:Any=None,
                       n:int=5):
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
        if dataset_type in ["math", "ocw"]:
            exec_func = extract_ans_from_cot_MATHnOCW 
        elif dataset_type == "gsm":
            exec_func = extract_num_turbo
        else:
            raise ValueError(f"dataset_type: {dataset_type} must be in 'gsm' 'math' 'ocw'")
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
        dnew, dold = check_answer(pred, answer)
        # key = "correct" if check_answer(pred, answer) else "incorrect"
        # res[key].append(sln)
        res[f"pred_v_gt_d.new_d.old"].append((pred,answer,dnew,dold))
    
    return res
        
    

def main():
    # 다음 둘에 fewshot에 활용된 question들이 달려있다.
    # ocw_MATH_prompts.math_cot.user 
    # ocw_MATH_prompts.ocw_cot.user
    ymlf = "../utils/ocw_MATH_prompts.yaml"
    # prompt_d = OmegaConf.load(ymlf)
    prompt_d = yaml.full_load(open(ymlf))

    math_questions = prompt_d["math_cot"]["user"]
    math_answers = prompt_d["math_cot"]["answers"]

    ocw_questions = prompt_d["ocw_cot"]["user"][1:]
    ocw_answers = prompt_d["ocw_cot"]["answers"][1:]

    # test
    normalize_symbolic_expression(ocw_answers[-2])
    for a in ocw_answers:
        print(a)
        print("new", str(normalize_symbolic_expression(a)))
        print("original", str(normalize_final_answer(a)))

    # inference_kwargs
    cot_kwargs = dict(
        # question: str, 
        dataset_type = "tobefilled", 
        temperature = 0.5, 
        backbone = "gpt4turbo", # "GPT4-1106", #"chatgpt0125",
        seed=None,
    )
    pal_kwargs = dict(
        temperature=0.7, 
        backbone = "gpt4turbo", # "GPT4-1106", #"chatgpt0125",
        seed=None,
    )
    p2c_kwargs = dict(
        # question: str,  
        # n=1,
        plan_temperature = 0.5,
        code_temperature = 0.7,
        backbone = "gpt4turbo", # "`GPT4`-1106", #"chatgpt0125",
        seed = None,
    )

    # dataset_types = ["math", "ocw"]
    dataset_types = ["ocw"]
    methods = ["cot", "pal", "p2c"]
    

    """
    result_dict
        ocw:
            questions: list
            answers: list
            cot:
                0: 
                    correct: list
                    incorrect: list
                1: 
                    correct: list
                    incorrect: list
                ...
            pal:
                ...
            p2c:
                ...
        math:
            ...
        (gsm):
            ...
            
    """

    result_dict = dict()
    for dstype in dataset_types:
        if dstype == "ocw":
            questions = ocw_questions
            answers = ocw_answers
        # elif dstype == "math":
        #     questions = math_questions
        #     answers = math_answers
        # elif dstype == "gsm":
        #     raise NotImplementedError("gsm questions and answers need to be loaded above")
        #     questions = gsm_questions
        #     answers = gsm_answers
        else:
            raise ValueError(f"dataset_type: {dstype} should be in ['ocw', 'math', 'gsm']")
        result_dict[dstype] = dict()
        for m in methods:
            result_dict[dstype][m] = dict()
            result_dict[dstype]["questions"] = questions
            result_dict[dstype]["answers"] = answers
            kwargs = eval(f"{m}_kwargs")
            if m == "cot":
                kwargs.update({"dataset_type": dstype})
            for i, q, a in zip(range(len(questions)), questions, answers):
                result_dict[dstype][m][i] = correct_incorrect_query_results(
                                                                question=q, 
                                                                answer=a,
                                                                method=m, 
                                                                dataset_type=dstype, 
                                                                inference_kwargs=kwargs, 
                                                                )    
    
    jsonf = "ocw_test_symexp_eval.json"
    with open(jsonf, "w") as f:
        json.dump(result_dict, f, indent=4)
        print(jsonf, "saved")

if __name__ == "__main__":
    Fire(main)
    
