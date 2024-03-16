from utils.math_util import math_check_answer
import jsonlines as jsl
from pqdm.processes import pqdm
from tqdm import tqdm

def complete_row(row):
    row["eval_new"] = math_check_answer(row["answer"], row["answer"])
    return row 


if __name__ == "__main__":
    records = list(jsl.open("edgecases_math_eval_not_working.jsonl"))
    records = pqdm(records, complete_row, n_jobs=8)
    # for row in tqdm(records):
    #     row = complete_row(row)
    with jsl.open("edgecases_math_eval_not_working.jsonl_fixed", "w") as f:
        f.write_all(records)
