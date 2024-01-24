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
    corrects = v_diff(df.majority_ans, df.answer).sum()
    return corrects


def eval_math(df):
    df["submission"] = df.majority_ans.astype("str")
    equiv_flag = df.apply(
        lambda row: is_equiv(
            normalize_final_answer(row["answer"]),
            normalize_final_answer(row["submission"]),
        ),
        axis=1,
    )

    return equiv_flag.sum()

def eval_ocw(df):
    df["submission"] = df.majority_ans.astype("str")
    equiv_flag = df.apply(
        lambda row: is_equiv_ocw(
            normalize_final_answer(row["answer"]),
            normalize_final_answer(row["submission"]),
        ),
        axis=1,
    )

    return equiv_flag.sum()


def main(eval_jslf: str, eval_type: str = Literal["gsm", "math"]):
    df = pd.DataFrame(jsl.open(eval_jslf))
    nonconflict_mask = df.selection_or_rims.apply(
        lambda d: d["majority_vote"] if "majority_vote" in d.keys() else False
    )
    fail_mask = df.selection_or_rims.apply(
        lambda d: d["error"] if "error" in d.keys() else False
    )  # api error
    conflict_mask = ~(nonconflict_mask | fail_mask)
    df_conflict_only = df[conflict_mask]
    df_nonconflict_only = df[nonconflict_mask]

    total = len(df)
    failcount = fail_mask.sum()

    if eval_type in ["gsm", "svamp"]:
        nonconf_correct = eval_gsm_svamp(df_nonconflict_only)
        conf_correct = eval_gsm_svamp(df_conflict_only)
    elif eval_type == "math":
        nonconf_correct = eval_math(df_nonconflict_only)
        conf_correct = eval_math(df_conflict_only)
    elif eval_type == "ocw":
        nonconf_correct = eval_ocw(df_nonconflict_only)
        conf_correct = eval_ocw(df_conflict_only)
    else:
        raise ValueError

    print(f"total: {nonconf_correct + conf_correct} / {total}")
    print(
        f"fail: {failcount} / {total}",
    )
    print(f"nonconflict: {nonconf_correct} / {nonconflict_mask.sum()}")
    print(f"conflict: {conf_correct} / {conflict_mask.sum()}")


if __name__ == "__main__":
    Fire(main)
