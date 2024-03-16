from utils.math_util import math_check_answer, ocw_check_answer
import jsonlines as jsl


matheval_fails_on = [
                    [
                        "$[2,5)$",
                        "[2,5)",
                        "incorrect"
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
]


def complete_row(row:dict, eval_f:callable=None)->dict:
    try:
        row[eval_f.__name__] = eval_f(row["pred"], row["answer"])
        # row["eval"] = eval_f(row["answer"], row["answer"])
    except Exception as e:
        row[eval_f.__name__] = f"EVAL_FAIL! {str(e)}"
    return row


records = [dict(zip(["pred", "answer", "initial_decision"], (p,a,d))) for p, a, d in matheval_fails_on]
for row in records:
    row = complete_row(row, eval_f = math_check_answer)
    row = complete_row(row, eval_f = ocw_check_answer)
with jsl.open("test_check_math_ans_cornercase.jsonl", "w") as f:
    f.write_all(records)
    print("test_check_math_ans_cornercase.jsonl is written.")