from utils.math_util import ocw_check_answer, math_check_answer # is_equiv_ocw, is_equiv

import jsonlines as jsl 
from tqdm import tqdm
import pandas as pd 
from pqdm.processes import pqdm

from functools import partial
from typing import Union


task2check_d = {"math":
    [
                    [
                        "$\\boxed{[2, 5)}$",
                        "[2,5)",
                        "incorrect"
                    ],
                    [
                        "$[2,5)$",
                        "[2,5)",
                        "incorrect"
                    ],
                    [
                        "$ in the interval $",
                        "[2,5)",
                        "incorrect"
                    ],
                    [
                        "$ is $",
                        "[2,5)",
                        "incorrect"
                    ],
                    [
                        "$\\boxed{[2, 5)}$",
                        "[2,5)",
                        "incorrect"
                    ],
                    [
                        "$\\boxed{[2, 5)}$",
                        "[2,5)",
                        "incorrect"
                    ],
                    [
                        "$[2, 5)$",
                        "[2,5)",
                        "incorrect"
                    ],
                    [
                        "$[2, 5)$",
                        "[2,5)",
                        "incorrect"
                    ],
                    [
                        "480",
                        16,
                        "incorrect"
                    ],
                    [
                        "$\nn = \\frac{480}{30} = 16\n$",
                        16,
                        "correct"
                    ],
                    [
                        "$n = 16$",
                        16,
                        "correct"
                    ],
                    [
                        "16",
                        16,
                        "correct"
                    ],
                    [
                        "$n = 16$",
                        16,
                        "correct"
                    ],
                    [ # not sure why the followings are not correct ("." after 16?)
                        "\\(n = 16.\\)",
                        16,
                        "incorrect"
                    ],
                    [
                        "\\[ n = 16 \\]",
                        16,
                        "incorrect"
                    ], # here too.
                    [
                        "$ is $",
                        "-\\frac{2}{3}",
                        "incorrect"
                    ],
                    [
                        "$\\boxed{-\\frac{2}{3}}$",
                        "-\\frac{2}{3}",
                        "correct"
                    ],
                    [
                        "$ is $",
                        "-\\frac{2}{3}",
                        "incorrect"
                    ],
                    [
                        "$\\frac{a}{b} = -\\frac{2}{3}$",
                        "-\\frac{2}{3}",
                        "correct"
                    ],
                      [
                        -0.6666666666666666, # normalizing function is not the same between two --> decided as unprocessable
                        "-\\frac{2}{3}",
                        "incorrect"
                    ],
                ],
    "ocw": [
            [
                "\\[ y(t) = \\frac{e^{-at} - e^{-bt}}{b-a} \\]",
                "\\[\\frac{1}{b-a}\\left(e^{-a t}-e^{-b t}\\right)\\]",
                "incorrect"
            ],
            [
                "$p(s) = s^2 + bs + 1$",
                "$s^{2}+b s+1$",
                "incorrect" # what?
            ],
            [
                "$s^2 + bs + 1$",
                "$s^{2}+b s+1$",
                "correct" # and what?
            ],
            [
                "\\( s^2 + bs + 1 \\)",
                "$s^{2}+b s+1$",
                "incorrect"
            ],
            [
                "\\( p(s) = s^2 + bs + 1 \\)",
                "$s^{2}+b s+1$",
                "correct"
            ],
            [
                "\\[\np(s) = s^2 + bs + 1\n\\]",
                "$s^{2}+b s+1$",
                "incorrect"
            ],
            [
                "\\( s^2 + bs + 1 \\)",
                "$s^{2}+b s+1$",
                "incorrect"
            ],
            [
                "\\[p(s) = s^2 + bs + 1\\]",
                "$s^{2}+b s+1$",
                "incorrect"
            ]
        ]
}


def complete_row(row:dict, eval_f:callable=None)->dict:
    try:
        row[eval_f.__name__] = eval_f(row["pred"], row["answer"])
        # row["eval"] = eval_f(row["answer"], row["answer"])
    except Exception as e:
        row[eval_f.__name__] = f"EVAL_FAIL! {str(e)}"
    return row

def isfail(eval:Union[bool, str])->bool:
    if isinstance(eval, bool):
        return False
    elif isinstance(eval, str):
        return eval.startswith("EVAL_FAIL!")

if __name__ == "__main__":
    for key, pred_ans_decide_records in task2check_d.items():
        records = [dict(zip(["pred", "answer", "initial_decision"],p_a_d)) for p_a_d in pred_ans_decide_records] 
        
        if key=="math":
            # eval_f = is_equiv
            eval_f = math_check_answer
        elif key=="ocw":
            # eval_f = is_equiv_ocw
            eval_f = ocw_check_answer
        else:
            raise ValueError(f"key: {key} is not supported")
        
        complete_row_ = partial(complete_row, eval_f=eval_f)
        records = pqdm(records, complete_row_, n_jobs=8)
        if key == "math":
            eval_f = ocw_check_answer # would ocw eval work better on math?
            complete_row__ = partial(complete_row, eval_f=eval_f)
            records = pqdm(records, complete_row__, n_jobs=8)

        newf = f"{key}_test_equiv_f_fewshots_cornercase.jsonl"
        with jsl.open(newf, "w") as writer:
            for row in records:
                try:
                    writer.write(row)
                except Exception as e:
                    print(e)
                    print(row)
            print(newf)

