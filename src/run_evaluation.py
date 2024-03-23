from typing import Literal

import jsonlines as jsl
import pandas as pd
from fire import Fire
import numpy as np

from utils.math_util import ocw_check_answer, math_check_answer, gsm_check_answer #is_equiv, is_equiv_ocw, normalize_final_answer,

from pprint import pprint
from collections import OrderedDict
from tqdm import tqdm
tqdm.pandas()


def eval_gsm_svamp(df, return_flag:bool=False, submission_col_already_exists:bool=False):
    if not submission_col_already_exists:
        df["submission"] = df.majority_ans.astype("str")
    equiv_flag = df.progress_apply(
        lambda row: gsm_check_answer(row.submission, row.answer), 
        axis=1
        )
    if return_flag:
        return equiv_flag
    else:
        return equiv_flag.sum() if len(df)>0 else 0

def eval_math(df, return_flag:bool=False, submission_col_already_exists:bool=False):
    if not submission_col_already_exists:
        df["submission"] = df.majority_ans.astype("str")
    equiv_flag = df.progress_apply(
        # lambda row: is_equiv(
        #     normalize_final_answer(row["submission"]),
        #     normalize_final_answer(row["answer"]),
        # ),
        # axis=1,
        lambda row: math_check_answer(row.submission, row.answer), axis=1)
    if return_flag:
        return equiv_flag
    else:
        return equiv_flag.sum() if len(df)>0 else 0

def eval_ocw(df, return_flag:bool=False, submission_col_already_exists:bool=False):
    if not submission_col_already_exists:
        df["submission"] = df.majority_ans.astype("str")
    equiv_flag = df.progress_apply(
        lambda row: ocw_check_answer(
            row["submission"],
            row["answer"],
        ),
        axis=1,
    )
    if return_flag:
        return equiv_flag if len(df)>0 else 0
    else:
        return equiv_flag.sum() if len(df)>0 else 0

def each_corrects(df, eval_type:Literal["gsm", "math", "ocw"], submission_col_already_exists:bool=False):
    if eval_type in "gsm svamp":
        evalf = eval_gsm_svamp
    elif eval_type == "ocw":
        evalf = eval_ocw
    elif eval_type == "math":
        evalf = eval_math
    else:
        assert False, "eval_type must be one of gsm, math, ocw, svamp"
    df["submission"] = df.ansmap.apply(lambda d: d["cot"])
    cot_corrects = evalf(df, return_flag=True, submission_col_already_exists=True)
    df["submission"] = df.ansmap.apply(lambda d: d["pal"])
    pal_corrects = evalf(df, return_flag=True, submission_col_already_exists=True)
    df["submission"] = df.ansmap.apply(lambda d: d["p2c"])
    p2c_corrects = evalf(df, return_flag=True, submission_col_already_exists=True)
    return cot_corrects, pal_corrects, p2c_corrects

def overlaps_corrects(cot, pal, p2c, return_flags:bool=False):
    cot, pal, p2c = map(lambda x: x.fillna(False), [cot, pal, p2c] )
    res = {
        "cot": cot,
        "pal": pal,
        "p2c": p2c,
        "all": (cot & pal & p2c),
        "cotpal-p2c": (cot & pal) & (~(cot & pal & p2c)), 
        "palp2c-cot": (pal & p2c) & (~(cot & pal & p2c)), 
        "p2ccot-pal": (p2c & cot) & (~(cot & pal & p2c)),
        "cot_only" : cot & (~pal) & (~p2c),
        "pal_only" : pal & (~cot) & (~p2c),
        "p2c_only" : p2c & (~cot) & (~pal),
    }
    if not return_flags:
        res = {k: v.sum() for k, v in res.items()}
    return res 

def main(skip_maineval:bool=True):
    base_rims_jslfs = OrderedDict({
        "gsm": (
            "/Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RESULTS_v1/gsm_0613long/chatgpt0613long_model_selection3_gsm.jsonl",
            "/Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RES_v3/gsm_0613long/chatgpt0613long_rims_gsm.jsonl",
            eval_gsm_svamp,
        ),
        "math": ( 
            "seonil_scripts/0_RESULTS_v1/math_full_0613long/chatgpt0613long_model_selection3_mathmerged.jsonl",
            "seonil_scripts/0_RES_v3/math_full_0613long/chatgpt0613long_rims_mathmerged.jsonl",
            eval_math,
        ),
        "ocw": (
            "seonil_scripts/0_RESULTS_v1/ocw_0613long/chatgpt0613long_model_selection3_ocw.jsonl", 
            "seonil_scripts/0_RES_v3/ocw_0613long/chatgpt0613long_rims_ocw.jsonl",
            eval_ocw,
        ),
        # "svamp":,
    })
    str2df = lambda txt: pd.DataFrame(jsl.open(txt))
    bases = {k: (None, str2df(v[0])) if skip_maineval else \
                (v[2](str2df(v[0])), (str2df(v[0]))) \
                    for k,v in base_rims_jslfs.items()}
    
    rims = {k: (None, str2df(v[1])) if skip_maineval else \
                (v[2](str2df(v[1])), (str2df(v[1]))) \
                    for k,v in base_rims_jslfs.items()}

    # major results (concordant answer effect by eval f update not applied here but...)
    print("# major results (*assuming majority vote results stays still (which is hardly true because evaluation, parsing, and execution has modified) )")
    print("this is results by previous prompts (gsm fewshots applied to ocw, math) and only evaluation function is modified")
    pprint({k:(v1, len(v2)) for k,(v1,v2) in bases.items()})
    pprint({k:(v1, len(v2)) for k,(v1,v2) in rims.items()})
    
    # overlaps counts
    print("# single over each dataset")
    for k, (corrects_, df) in bases.items():
        print(f"## {k}")
        cot_correct, pal_correct, p2c_correct = each_corrects(df, eval_type=k)
        overlaps_dict = overlaps_corrects(cot_correct, pal_correct, p2c_correct)
        pprint(overlaps_dict)
    


if __name__ == "__main__":
    Fire(main)
