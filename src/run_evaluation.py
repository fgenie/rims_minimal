from typing import Literal

import jsonlines as jsl
import pandas as pd
from fire import Fire
import numpy as np

from utils.math_util import is_equiv, is_equiv_ocw, normalize_final_answer

from pprint import pprint
from collections import OrderedDict

def eval_gsm_svamp(df, return_flag:bool=False, use_submission:bool=False):
    if not use_submission:
        df["submission"] = df.majority_ans.astype("str")
    def diff_if_possible(a1, a2):
        try:
            return abs(a1-a2) < 1e-3
        except:
            return False 
    v_diff = np.vectorize(diff_if_possible)
    corrects = v_diff(df.majority_ans, df.answer)
    if return_flag:
        return corrects if len(df)>0 else 0
    else:
        return corrects.sum() if len(df)>0 else 0

def eval_math(df, return_flag:bool=False, use_submission:bool=False):
    if not use_submission:
        df["submission"] = df.majority_ans.astype("str")
    equiv_flag = df.apply(
        lambda row: is_equiv(
            normalize_final_answer(row["answer"]),
            normalize_final_answer(row["submission"]),
        ),
        axis=1,
    )
    if return_flag:
        return equiv_flag
    else:
        return equiv_flag.sum() if len(df)>0 else 0

def eval_ocw(df, return_flag:bool=False, use_submission:bool=False):
    if not use_submission:
        df["submission"] = df.majority_ans.astype("str")
    equiv_flag = df.apply(
        lambda row: is_equiv_ocw(
            normalize_final_answer(row["answer"]),
            normalize_final_answer(row["submission"]),
        ),
        axis=1,
    )
    if return_flag:
        return equiv_flag if len(df)>0 else 0
    else:
        return equiv_flag.sum() if len(df)>0 else 0

def each_corrects(df, eval_type:Literal["gsm", "math", "ocw"], use_submission:bool=False):
    if eval_type in "gsm svamp":
        evalf = eval_gsm_svamp
    elif eval_type == "ocw":
        evalf = eval_ocw
    elif eval_type == "math":
        evalf = eval_math
    else:
        assert False, "eval_type must be one of gsm, math, ocw, svamp"
    df["submission"] = df.ansmap.apply(lambda d: d["cot"])
    cot_corrects = evalf(df, return_flag=True, use_submission=True)
    df["submission"] = df.ansmap.apply(lambda d: d["pal"])
    pal_corrects = evalf(df, return_flag=True, use_submission=True)
    df["submission"] = df.ansmap.apply(lambda d: d["p2c"])
    p2c_corrects = evalf(df, return_flag=True, use_submission=True)
    return cot_corrects, pal_corrects, p2c_corrects

def overlaps_corrects(cot, pal, p2c, return_flags:bool=False):
    res = {
        "cot": cot,
        "pal": pal,
        "p2c": p2c,
        "cotpal": (cot & pal), 
        "palp2c": (pal & p2c), 
        "p2ccot": (p2c & cot),
        "all": (cot & pal & p2c)
    }
    if not return_flags:
        res = {k: v.sum() for k, v in res.items()}
    return res 

def main():
    base_rims_jslfs = OrderedDict({
        "gsm": (
            "/Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RESULTS_v1/gsm_0613long/chatgpt0613long_model_selection3_gsm.jsonl",
            "/Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RES_v3/gsm_0613long/chatgpt0613long_rims_gsm.jsonl"
        ),
        "math": ( 
            "0_RESULTS_v1/math_full_0613long/chatgpt0613long_model_selection3_mathmerged.jsonl",
            "0_RES_v3/math_full_0613long/chatgpt0613long_rims_mathmerged.jsonl"
        ),
        "ocw": (
            "0_RESULTS_v1/ocw_0613long/chatgpt0613long_model_selection3_ocw.jsonl", 
            "0_RES_v3/ocw_0613long/chatgpt0613long_rims_ocw.jsonl"
        ),
        # "svamp":,
    })
    str2df = lambda txt: pd.DataFrame(jsl.open(txt))
    bases = {k:str2df(v[0]) for k,v in base_rims_jslfs.items()}
    rims = {k:str2df(v[1]) for k,v in base_rims_jslfs.items()}
    
    # overlaps counts
    print("# single over each dataset")
    for k, df in bases.items():
        print(f"## {k}")
        cot_correct, pal_correct, p2c_correct = each_corrects(df, eval_type=k)
        overlaps_dict = overlaps_corrects(cot_correct, pal_correct, p2c_correct)
        pprint(overlaps_dict)
    


if __name__ == "__main__":
    Fire(main)
