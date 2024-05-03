# evaluation script for self-consistency setting
# i.e. jsonlines file fields differ from n==1 case


# will check: majority_ans == GT answer
"""
simple greedy
            # update row: need to consider later it will be reused for rims inferencing.
            row["error"] = False
            row["error_msg"] = ""
            row["runnning_at"] = "baseline_complete_row"

            row["majority_ans"] = majority_ans
            row["idx2chosen_method"] = idx2chosen_method
            row["majvote_ans"] = majvote_ans
            row["candid_answers"] = candid_answers  # for debug use
            row["inference_mode"] = [
                "majority_vote" if majvote_ans is not None else "selection"
                for majvote_ans in majvote_ans
            ]
            row["dataset_type"] = dataset_type
            row["prompt_file"] = prompt_f
            row["temperatures"] = {
                "cot_temperature": 0.5,
                "pal_temperature": 0.8,
                "n": n,
            }


rims
            # update row
            row["error"] = False
            row["error_msg"] = ""
            row["running_at"] = "rims_complete_row"

            row["majority_ans"] = majority_ans
            row["idx2chosen_method"] = idx2chosen_method
            # row["majvote_ans"] = majvote_ans # not changed
            row["candid_answers"] = candid_answers
            row["inference_mode"] = ["majority_vote" if majvote_ans is not None else "rims" for majvote_ans in majvote_ans]
            row["dataset_type"] = dataset_type # for logging use... overwrite the dataset_type
            row["prompt_file"] = str(prompt_f)
            row["temperatures"].update(
                {"rims_temperature": temperature, "n": n, "n_adj": n_adj}
            )
            row["rims_query_results"] = eval_friendly_d_ # aggregated rims results
"""

# according to the fields above


# for math and ocw
#   majority_ans: List (length 1 or 2)
# if 2, check if any of those are correct (considers correct if any of them are correct)


from pathlib import Path
from typing import Literal

import jsonlines as jsl
import numpy as np
import pandas as pd
from fire import Fire
from tqdm import tqdm

from utils.math_util import gsm_check_answer, math_check_answer, ocw_check_answer

tqdm.pandas()


def eval_gsm_svamp(
    df, return_flag: bool = False, submission_col_already_exists: bool = False
):
    if not submission_col_already_exists:
        df["submission"] = df.majority_ans
    df.submission = df.submission.astype("str")
    equiv_flag = df.progress_apply(
        lambda row: gsm_check_answer(row.submission, row.answer)
        if not isinstance(row.submission, list)
        else (  # this part is for n>1 scenario. math/ocw will return top-2 candids if they are joint 1st-place.
            gsm_check_answer(row.submission[0], row.answer)
            or gsm_check_answer(row.submission[1], row.answer)
        ),
        axis=1,
    )

    if return_flag:
        return equiv_flag
    else:
        return equiv_flag.sum() if len(df) > 0 else 0


def list_apply_check(checkf, answer, submission):
    if isinstance(submission, list):
        return any([checkf(sub, answer) for sub in submission])
    else:
        return checkf(submission, answer)


def eval_math(
    df, return_flag: bool = False, submission_col_already_exists: bool = False
):
    if not submission_col_already_exists:
        df["submission"] = df.majority_ans
    # df.submission = df.submission.astype("str")
    df.submission = df.submission.apply(
        lambda x: [str(xx) for xx in x] if isinstance(x, list) else str(x)
    )
    equiv_flag = df.progress_apply(
        lambda row: list_apply_check(math_check_answer, row.answer, row.submission),
        # lambda row: math_check_answer(row.submission, row.answer)
        # if not isinstance(row.submission, list)
        # else (  # this part is for n>1 scenario. math/ocw will return top-2 candids if they are joint 1st-place.
        #     math_check_answer(row.submission[0], row.answer)
        #     or math_check_answer(row.submission[1], row.answer)
        # ),
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
    df.submission = df.submission.apply(
        lambda x: [str(xx) for xx in x] if isinstance(x, list) else str(x)
    )
    equiv_flag = df.progress_apply(
        lambda row: list_apply_check(ocw_check_answer, row.answer, row.submission),
        # lambda row: ocw_check_answer(row.submission, row.answer)
        # if not isinstance(row.submission, list)
        # else (  # this part is for n>1 scenario. math/ocw will return top-2 candids if they are joint 1st-place.
        #     ocw_check_answer(row.submission[0], row.answer)
        #     or ocw_check_answer(row.submission[1], row.answer)
        # ),
        axis=1,
    )
    if return_flag:
        return equiv_flag if len(df) > 0 else 0
    else:
        return equiv_flag.sum() if len(df) > 0 else 0


def main(
    ptn: str = "outputs/ocw_course_dt.ocw/gpt4turbo/*/*jsonl",
    eval_type: Literal["gsm", "math", "ocw", "svamp"] = "ocw",
    outf: str = "testout.txt",
):
    # get jsonl files
    paths = list(Path().glob(ptn))
    for jslf in paths:
        # load data
        df = pd.DataFrame(jsl.open(jslf))
        df = df.drop_duplicates(
            subset=["problem" if eval_type == "ocw" else "question"]
        )

        # logfile open
        f = open(outf, "a")

        eval_type2eval_f = {
            "gsm": eval_gsm_svamp,
            "math": eval_math,
            "ocw": eval_ocw,
        }

        eval_f = eval_type2eval_f[eval_type]
        total = len(df)

        # performance overall
        # overall acc. / success rate base_rims / num(maj, (base or rims), failed)
        print("=======overall performance=======", file=f)
        fail_mask = df.error
        df = df[~fail_mask]
        majority_vote_mask = df.majvote_ans.apply(lambda lst: lst.count(None) == 0)

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
        successes = (corrects_mask & (~majority_vote_mask)).sum(), total - num_maj
        print(f"file    = {jslf}", file=f)
        print(f"dataset = {eval_type} ({total} rows)", file=f)

        print(
            f"overall_acc:\t{corrects[0]:<5}/ {total:>4} ({corrects[0]/total*100:.1f}%)",
            file=f,
        )
        print(
            f"success_rate:\t{successes[0]:<5}/ {successes[1]:>4} ({successes[0]/successes[1]*100:.1f}%)",
            file=f,
        )
        print(
            f"{total} (total) = \n\t{total-num_maj} (seleciton) \n\t+ {num_maj} (maj-votes) \n\t+ {num_fails} (fails: counted as incorrect)",
            file=f,
        )
        print("=====================================\n\n", file=f)

        f.close()


if __name__ == "__main__":
    Fire(main)
