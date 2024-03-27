from typing import Literal

import jsonlines as jsl
import pandas as pd
from fire import Fire
import numpy as np
from pathlib import Path

from utils.math_util import math_check_answer, ocw_check_answer, gsm_check_answer   
from tqdm import tqdm

tqdm.pandas()


# eval functions with exception handled (like len(df)==0)
def eval_gsm_svamp(df, 
                   return_flag:bool=False, 
                   submission_col_already_exists:bool=False):
    if not submission_col_already_exists:
        df["submission"] = df.majority_ans
    df.submission = df.submission.astype("str")
    equiv_flag = df.progress_apply(
        lambda row: gsm_check_answer(row.submission, row.answer), 
        axis=1
        )
    if return_flag:
        return equiv_flag
    else:
        return equiv_flag.sum() if len(df)>0 else 0

def eval_math(df, 
              return_flag:bool=False, 
              submission_col_already_exists:bool=False):
    if not submission_col_already_exists:
        df["submission"] = df.majority_ans
    df.submission = df.submission.astype("str")
    equiv_flag = df.progress_apply(
        lambda row: math_check_answer(row.submission, row.answer), axis=1)
    if return_flag:
        return equiv_flag
    else:
        return equiv_flag.sum() if len(df)>0 else 0

def eval_ocw(df, 
             return_flag:bool=False, 
             submission_col_already_exists:bool=False):
    if not submission_col_already_exists:
        df["submission"] = df.majority_ans
    df.submission = df.submission.astype("str")
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




def overlaps_corrects(cot, pal, p2c, return_flags:bool=False):
    cot, pal, p2c = map(lambda x: x.fillna(False), [cot, pal, p2c] )
    res = {
        # "cot": cot,
        # "pal": pal,
        # "p2c": p2c,
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


def main(
        eval_jslf: str = "outputs/MATH-full_dt.math/chatgpt0613long/model_selection_prompts/merged.jsonl", 
        eval_type: Literal["gsm", "math", "ocw", "svamp"] = "math", 
        outf:str = "math_baseline.txt",
        eval_indiv_and_overlap: bool = False,       
        ):

    # load data
    df = pd.DataFrame(jsl.open(eval_jslf))
    
    # logfile open
    f = open(outf, 'a')

    eval_type2eval_f = {
    "gsm": eval_gsm_svamp,
    "math": eval_math,
    "ocw": eval_ocw,
    }

    eval_f = eval_type2eval_f[eval_type]

    # indiv performance
    total = len(df)
    if eval_indiv_and_overlap:
        each_corrects = dict()
        print("=======individual performance=======", file=f)
        for method in "cot pal p2c".split():
            df["submission"] = df.ansmap.apply(lambda d: d[method] if isinstance(d, dict) else None)
            corrects_mask_ = eval_f(df, return_flag=True, submission_col_already_exists=True)
            print(f"file      = {eval_jslf}", file=f)
            print(f"dataset   = {eval_type}", file=f)
            print(f"{method}:\t{corrects_mask_.sum():<5}/ {total:>4} ({corrects_mask_.sum()/total*100:.1f}%)", file=f)
            print("\n", file=f)
            each_corrects[method] = corrects_mask_
        print("=====================================\n\n", file=f)

        # overlaps
        overlaps = overlaps_corrects(*each_corrects.values())
        print("========distinctive behaviors========", file=f)
        print(str(overlaps), file=f)

        print("=====================================\n\n", file=f)

        
    # performance overall
    # overall acc. / success rate base_rims / num(maj, (base or rims), failed)
    print("=======overall performance=======", file=f)
    fail_mask = df.selection_or_rims.apply(
        lambda d: d["error"] if "error" in d.keys() else False
    )  # api error
    majority_vote_mask = df.selection_or_rims.apply(
        lambda d: d["majority_vote"] if "majority_vote" in d.keys() else False
    )

    corrects_mask = eval_f(df, return_flag=True)
    
    # fillna first to avoid error 
    fail_mask, majority_vote_mask, corrects_mask = map(
        lambda x: x.fillna(False), [fail_mask, majority_vote_mask, corrects_mask]
    ) 

    # overall acc.
    total = len(df)
    num_maj = majority_vote_mask.sum()    
    num_fails = fail_mask.sum()
    corrects = corrects_mask.sum(), total
    successes = (corrects_mask & (~majority_vote_mask)).sum(),  total-num_maj
    print(f"file    = {eval_jslf}", file=f)
    print(f"dataset = {eval_type} ({total} rows)", file=f)
    
    print(f"overall_acc:\t{corrects[0]:<5}/ {total:>4} ({corrects[0]/total*100:.1f}%)", file=f)
    print(f"success_rate:\t{successes[0]:<5}/ {successes[1]:>4} ({successes[0]/successes[1]*100:.1f}%)", file=f)
    print(f"{total} (total) = \n\t{total-num_maj} (seleciton) \n\t+ {num_maj} (maj-votes) \n\t+ {num_fails} (fails: counted as incorrect)", file=f)
    print("=====================================\n\n", file=f)
    
    f.close()




if __name__ == "__main__":
    Fire(main)
