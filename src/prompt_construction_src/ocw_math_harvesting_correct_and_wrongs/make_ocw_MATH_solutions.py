from utils.llm_query_utils import query_cot, query_pal, query_plancode, _query, \
                                safe_execute_turbo, extract_num_turbo, extract_ans_from_cot_MATHnOCW, backbone2model
from utils.math_util import is_equiv, \
                            is_equiv_ocw, \
                            normalize_final_answer

from fire import Fire 
# from omegaconf import OmegaConf 
import yaml
import json


from functools import partial
from typing import Callable, Literal, Any
from collections import defaultdict
from tqdm import tqdm
from utils.math_util import ocw_check_answer, math_check_answer, gsm_check_answer



def correct_incorrect_query_results(question:str="", 
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
        if dataset_type == "gsm":
            exec_func = extract_num_turbo
        else: # ocw, math
            exec_func = extract_ans_from_cot_MATHnOCW
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
        res[f"pred_v_gt_decision"].append((pred,answer,key))
    
    return res
        
    

def main():
    # 다음 둘에 fewshot에 활용된 question들이 달려있다.
    # ocw_MATH_prompts.math_cot.user 
    # ocw_MATH_prompts.ocw_cot.user
    ymlf = "../utils/ocw_MATH_prompts.yaml"
    # prompt_d = OmegaConf.load(ymlf)
    prompt_d = yaml.full_load(open(ymlf))

    math_questions = prompt_d["math_cot"]["questions"]
    math_answers = prompt_d["math_cot"]["answers"]

    ocw_questions = prompt_d["ocw_cot"]["questions"]
    ocw_answers = prompt_d["ocw_cot"]["answers"]

    # test
    # normalize_symbolic_expression(ocw_answers[-2])
    # for a in ocw_answers:
    #     print(a)
    #     print("new", str(normalize_symbolic_expression(a)))
    #     print("original", str(normalize_final_answer(a)))

    # inference_kwargs
    cot_kwargs = dict(
        # question: str, 
        dataset_type = "tobefilled", 
        temperature = 1., 
        backbone = "gpt4turbo", 
        seed=None,
    )
    pal_kwargs = dict(
        dataset_type = "tobefilled", 
        temperature=1., 
        backbone = "gpt4turbo",
        seed=None,
    )
    p2c_kwargs = dict(
        # question: str,  
        # n=1,
        plan_temperature = 2.,
        code_temperature = 2., # to get the wrong ones!
        backbone = "chatgpt0125", #  
        seed = None,
    )

    dataset_types = ["math", "ocw"]
    # methods = ["cot", "pal", "p2c"]
    methods = ["p2c"]
    

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
        elif dstype == "math":
            questions = math_questions
            answers = math_answers
        elif dstype == "gsm":
            raise NotImplementedError("gsm questions and answers need to be loaded above")
            questions = gsm_questions
            answers = gsm_answers
        else:
            raise ValueError(f"dataset_type: {dstype} should be in ['ocw', 'math', 'gsm']")
        result_dict[dstype] = dict()
        for m in methods:
            result_dict[dstype][m] = dict()
            result_dict[dstype]["questions"] = questions
            result_dict[dstype]["answers"] = answers
            kwargs = eval(f"{m}_kwargs")
            if m in "cot pal":
                kwargs.update({"dataset_type": dstype})
            for i, q, a in tqdm(zip(range(len(questions)), questions, answers), total = len(questions), desc = f"{dstype} / {m}"):
                result_dict[dstype][m][i] = correct_incorrect_query_results(
                                                                question=q, 
                                                                answer=a,
                                                                method=m, 
                                                                dataset_type=dstype, 
                                                                inference_kwargs=kwargs, 
                                                                )    
    
    jsonf = "p2c_chatgpt_hightemp_mar18_1.json"
    with open(jsonf, "w") as f:
        json.dump(result_dict, f, indent=4)
        print(jsonf, "saved")
    
    # cost printout
    model = backbone2model("gpt4turbo")
    _query.print_summary()
    _query.tokens2usd(model)

if __name__ == "__main__":
    Fire(main)
    
