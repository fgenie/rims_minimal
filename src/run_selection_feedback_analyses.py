from pathlib import Path
from typing import List, Literal

import jsonlines as jsl
import numpy as np
import pandas as pd
from fire import Fire
from tqdm import tqdm

from utils.math_util import gsm_check_answer, math_check_answer, ocw_check_answer

tqdm.pandas()


# eval functions with exception handled (like len(df)==0)
def eval_gsm_svamp(
    df, return_flag: bool = False, submission_col_already_exists: bool = False
):
    if not submission_col_already_exists:
        df["submission"] = df.majority_ans
    df.submission = df.submission.astype("str")
    equiv_flag = df.progress_apply(
        lambda row: gsm_check_answer(row.submission, row.answer), axis=1
    )
    if return_flag:
        return equiv_flag
    else:
        return equiv_flag.sum() if len(df) > 0 else 0


def eval_math(
    df, return_flag: bool = False, submission_col_already_exists: bool = False
):
    if not submission_col_already_exists:
        df["submission"] = df.majority_ans
    df.submission = df.submission.astype("str")
    equiv_flag = df.progress_apply(
        lambda row: math_check_answer(row.submission, row.answer), axis=1
    )
    if return_flag:
        return equiv_flag
    else:
        return equiv_flag.sum() if len(df) > 0 else 0


def eval_ocw(
    df, return_flag: bool = False, submission_col_already_exists: bool = False
):
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
        return equiv_flag if len(df) > 0 else 0
    else:
        return equiv_flag.sum() if len(df) > 0 else 0


def separate_the_effects(
    cot, pal, p2c, successes: List[bool], selection_methods: List[str]
):
    cot, pal, p2c = map(lambda x: x.fillna(False), [cot, pal, p2c])

    # init
    selection_effect = pd.Series(False, index=successes.index)
    for method in "cot pal p2c".split():
        # if the selected method is correct originally
        # and the correct rims result selected the same method then it's a selection effect!
        selection_effect |= successes & (selection_methods == method) & eval(method)
    reflection_effect = ~selection_effect & successes

    res = {
        "reflection_effect": reflection_effect,
        "selection_effect": selection_effect,
        "select_p2c_success": selection_effect & (selection_methods == "p2c"),
        "select_pal_success": selection_effect & (selection_methods == "pal"),
        "select_cot_success": selection_effect & (selection_methods == "cot"),
    }

    # normalize the results
    for k in res.keys():
        res[
            k
        ] = f"{res[k].sum():>4} / {successes.sum():>4} ({100*res[k].sum()/successes.sum():<4.1f} %)"

    return res


def main(
    ptn: str = "outputs/gsm8K_test_dt.gsm/gpt4turbo/*/*jsonl",
    eval_type: Literal["gsm", "math", "ocw", "svamp"] = "gsm",
    outf: str = "testout.txt",
):
    # get jsonl files
    paths = list(Path().glob(ptn))

    table_idxs = []
    table_rows = []

    for jslf in paths:
        # table idx
        table_idxs.append(jslf.parent.name)

        # load data
        df = pd.DataFrame(jsl.open(jslf))

        # logfile open
        f = open(outf, "a")

        eval_type2eval_f = {
            "gsm": eval_gsm_svamp,
            "math": eval_math,
            "ocw": eval_ocw,
        }

        eval_f = eval_type2eval_f[eval_type]

        # performance overall
        # overall acc. / success rate base_rims / num(maj, (base or rims), failed)
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

        successes = corrects_mask & (~majority_vote_mask)
        selection_methods = df.selection_or_rims.apply(
            lambda d: d["good_method"] if "good_method" in d.keys() else None
        )

        # single methods corrects
        each_corrects = dict()
        for method in "cot pal p2c".split():
            df["submission"] = df.ansmap.apply(
                lambda d: d[method] if isinstance(d, dict) else None
            )
            corrects_mask_ = eval_f(
                df, return_flag=True, submission_col_already_exists=True
            )
            each_corrects[method] = corrects_mask_

        cot = each_corrects["cot"]
        pal = each_corrects["pal"]
        p2c = each_corrects["p2c"]

        analysis_row = separate_the_effects(
            cot[successes],
            pal[successes],
            p2c[successes],
            successes,
            selection_methods[successes],
        )

        # append to table
        table_rows.append(analysis_row)

    table = pd.DataFrame(table_rows, index=table_idxs).to_markdown()
    if not Path(outf).exists():
        with open(outf, "w") as w:
            w.write("\n")
    with open(outf, "a") as f:
        print(eval_type, file=f)
        print(model := jslf.parent.parent.name, file=f)
        print(table, file=f)


if __name__ == "__main__":
    Fire(main)
