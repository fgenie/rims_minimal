



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
    equiv_flag_new = df.apply(
        lambda row: is_equiv_ocw(
            row["answer"],
            row["submission"],
            use_sym_exp_normalizer=True
        ),
        axis=1,
    )
    equiv_flag_old = df.apply(
        lambda row: is_equiv_ocw(
            row["answer"],
            row["submission"],
            use_sym_exp_normalizer=False
        ),
        axis=1,
    )

    return equiv_flag_new, equiv_flag_old


def main(eval_jslf: str="/Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RES_v3/ocw_0613long/chatgpt0613long_rims_ocw.jsonl", eval_type:Literal["gsm", "math", "ocw", "svamp"]="ocw"):
    df = pd.DataFrame(jsl.open(eval_jslf))
    fail_mask = df.selection_or_rims.apply(
        lambda d: d["error"] if "error" in d.keys() else False
    )  # api error

    total = len(df)
    failcount = fail_mask.sum()
    for method in "cot pal p2c".split():
        df[f"submission"] = df.ansmap.apply(lambda d: d[method])
        # if eval_type in "gsm svamp".split():
        #     corrects = eval_gsm_svamp(df)
        # elif eval_type == "math":
        #     corrects = eval_math(df)
        if eval_type == "ocw":
            newcorr, oldcorr = eval_ocw(df)
        else:
            raise ValueError(f"eval_type {eval_type} not supported")
        with open('ocw_eval_oldnnew.txt', 'a') as f:
            print(f"### comparing old and new", file=f)
            print(f"{newcorr.sum()}, {oldcorr.sum()}  / {total}", file=f)
            # print(f"api fail: {failcount})")
        mask = (newcorr != oldcorr)
        df_ = df[mask]
        df_[method] = df[mask]["submission"]
        df_["new_old_iscorrect"] = list(*zip(newcorr[mask], oldcorr[mask]))
        df_ = df_.loc[:, ["answer", "submission", "new_old_iscorrect"]]
        df_.to_dict(orient="records")
        with jsl.open(f"ocw_{method}_oldnnew.jsonl", "w") as wd:
            wd.write_all(df_.to_dict(orient="records"))
        print(f"ocw_{method}_oldnnew.jsonl written!")        



if __name__ == "__main__":
    Fire(main)
