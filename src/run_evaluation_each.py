from typing import Literal

import jsonlines as jsl
import pandas as pd
from fire import Fire
import numpy as np

from utils.math_util import is_equiv, is_equiv_ocw, normalize_final_answer


def eval_gsm_svamp(df):
    def diff_if_possible(a1, a2):
        try:
            return abs(a1-a2) < 1e-3
        except:
            return False 
    v_diff = np.vectorize(diff_if_possible)
    corrects = v_diff(df.submission, df.answer).sum()
    return corrects


def eval_math(df):
    # df["submission"] = df.majority_ans.astype("str")
    df.submission = df.submission.astype("str")
    equiv_flag = df.apply(
        lambda row: is_equiv(
            normalize_final_answer(row["answer"]),
            normalize_final_answer(row["submission"]),
        ),
        axis=1,
    )

    return equiv_flag.sum()

def eval_ocw(df):
    # df["submission"] = df.majority_ans.astype("str")
    df.submission = df.submission.astype("str")
    equiv_flag = df.apply(
        lambda row: is_equiv_ocw(
            normalize_final_answer(row["answer"]),
            normalize_final_answer(row["submission"]),
        ),
        axis=1,
    )

    return equiv_flag.sum()


def main(eval_jslf: str, eval_type: str = Literal["gsm", "math", "ocw", "svamp"]):
    df = pd.DataFrame(jsl.open(eval_jslf))
    fail_mask = df.selection_or_rims.apply(
        lambda d: d["error"] if "error" in d.keys() else False
    )  # api error

    total = len(df)
    failcount = fail_mask.sum()
    for method in "cot pal p2c".split():
        df["submission"] = df.ansmap.apply(lambda d: d[method])
        if eval_type in "gsm svamp".split():
            corrects = eval_gsm_svamp(df)
        elif eval_type == "math":
            corrects = eval_math(df)
        elif eval_type == "ocw":
            corrects = eval_ocw(df)
        else:
            raise ValueError(f"eval_type {eval_type} not supported")
        
        print(f"### {method}")
        print(f"{corrects}  / {total} ({corrects/total*100:.1f}%)")
        # print(f"api fail: {failcount})")
        



if __name__ == "__main__":
    Fire(main)
