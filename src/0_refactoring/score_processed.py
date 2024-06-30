from pathlib import Path
from typing import Literal

import jsonlines as jsl
import numpy as np
import pandas as pd
from fire import Fire
from processings.math_util import gsm_check_answer, math_check_answer, ocw_check_answer
from tqdm import tqdm

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
        lambda row: math_check_answer(row.submission, row.answer)
        if not isinstance(row.submission, list)
        else (  # this part is for n>1 scenario. math/ocw will return top-2 candids if they are joint 1st-place.
            math_check_answer(row.submission[0], row.answer)
            or math_check_answer(row.submission[1], row.answer)
        ),
        axis=1,
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
        lambda row: ocw_check_answer(row.submission, row.answer)
        if not isinstance(row.submission, list)
        else (  # this part is for n>1 scenario. math/ocw will return top-2 candids if they are joint 1st-place.
            ocw_check_answer(row.submission[0], row.answer)
            or ocw_check_answer(row.submission[1], row.answer)
        ),
        axis=1,
    )
    if return_flag:
        return equiv_flag if len(df) > 0 else 0
    else:
        return equiv_flag.sum() if len(df) > 0 else 0


def overlaps_corrects(cot, pal, p2c, return_flags: bool = False):
    cot, pal, p2c = map(lambda x: x.fillna(False), [cot, pal, p2c])
    res = {
        # "cot": cot,
        # "pal": pal,
        # "p2c": p2c,
        "all": (cot & pal & p2c),
        "cotpal-p2c": (cot & pal) & (~(cot & pal & p2c)),
        "palp2c-cot": (pal & p2c) & (~(cot & pal & p2c)),
        "p2ccot-pal": (p2c & cot) & (~(cot & pal & p2c)),
        "cot_only": cot & (~pal) & (~p2c),
        "pal_only": pal & (~cot) & (~p2c),
        "p2c_only": p2c & (~cot) & (~pal),
    }
    if not return_flags:
        res = {k: v.sum() for k, v in res.items()}
    return res


def score_indiv(
    ptn: str = "dbg_llama/processed_indiv.jsonl",  # regex allowed
    eval_type: Literal["gsm", "math", "ocw", "svamp"] = "gsm",
    dataset_jsl: str = "../../dataset/gsm8K_test.jsonl",
):
    # get jsonl files
    paths = list(Path().glob(ptn))
    for jslf in paths:
        outpath = paths[0].parent / f"{paths[0].stem}_scored.txt"
        # load data
        df = pd.DataFrame(jsl.open(jslf))
        answers = pd.read_json(dataset_jsl, lines=True, orient="records").answer
        df["answer"] = answers
        assert len(df) == len(
            answers
        ), f"{ptn} has missing rows! ({len(df)!=len(answers)})"

        # logfile open
        f = open(outpath, "a")

        eval_type2eval_f = {
            "gsm": eval_gsm_svamp,
            "math": eval_math,
            "ocw": eval_ocw,
        }

        eval_f = eval_type2eval_f[eval_type]

        # indiv performance
        total = len(df)
        each_corrects = dict()
        print(f"file      = {jslf}", file=f)
        print(f"dataset   = {eval_type}", file=f)
        print("=======individual performance=======", file=f)
        for method in "cot pal p2c".split():
            submission = df[f"{method}_preds"]
            if df[f"{method}_preds"].apply(len).max() > 1:
                raise NotImplementedError()
            else:  # n==1
                df["submission"] = submission.apply(lambda x: x[0])
            corrects_mask_ = eval_f(
                df, return_flag=True, submission_col_already_exists=True
            )

            print(
                f"{method}:\t{corrects_mask_.sum():<5}/ {total:>4} ({corrects_mask_.sum()/total*100:.1f}%)",
                file=f,
            )
            print("\n", file=f)
            each_corrects[method] = corrects_mask_

        df_cor = pd.DataFrame(each_corrects)
        df = pd.concat([df, df_cor], axis="columns")
        df.to_json(
            Path(ptn).parent / Path(ptn).name.replace("jsonl", "_inspect.jsonl"),
            lines=True,
            orient="records",
        )
        scoredf = pd.DataFrame(
            {
                k: f"{v.sum()}/{len(v)} ({100*v.sum()/len(v):.1f} %)"
                for k, v in each_corrects.items()
            },
            index=[0],
        )
        scoredf.to_markdown(outpath.with_suffix(".md"))
        print("=====================================\n\n", file=f)

        # overlaps
        overlaps = overlaps_corrects(*each_corrects.values())
        print("========distinctive behaviors========", file=f)
        print(str(overlaps), file=f)

        print("=====================================\n\n", file=f)
        f.close()


def score_selected_result():
    raise NotImplementedError()


if __name__ == "__main__":
    Fire()
    """python score_processed.py score_indiv """