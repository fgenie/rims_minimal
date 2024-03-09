from utils.math_util import is_equiv_ocw, is_equiv

import jsonlines as jsl 
from tqdm import tqdm
import pandas as pd 
from pqdm.processes import pqdm

from functools import partial
from typing import Union

def complete_row(row:dict, eval_f:callable=None, key:str=None)->dict:
    row["artificial_wrong"] =  row["answer"] + "+1"
    try:
        row["eval"] = eval_f(row["artificial_wrong"], row["answer"])
        # row["eval"] = eval_f(row["answer"], row["answer"])
    except Exception as e:
        row["eval"] = f"EVAL_FAIL! {str(e)}"
    if key == "ocw": # if ocw, try another eval method 
        try:
            row["eval_new"] = eval_f(row["artificial_wrong"], row["answer"], approach_w_symexp=True)
            # row["eval_new"] = eval_f(row["answer"], row["answer"], approach_w_symexp=True)
        except Exception as e:
            row["eval_new"] = f"EVAL_FAIL! {str(e)}"
    return row

def isfail(eval:Union[bool, str])->bool:
    if isinstance(eval, bool):
        return False
    elif isinstance(eval, str):
        return eval.startswith("EVAL_FAIL!")

if __name__ == "__main__":

    records_d = dict (
        ocw = list(jsl.open("/Users/seonils/dev/rims_minimal/dataset/ocw/ocw_course.jsonl")), 
        # math = list(jsl.open("/Users/seonils/dev/rims_minimal/dataset/MATH/MATH-full.jsonl")), 
    )

    for key, records in records_d.items():
        if key=="math":
            eval_f = is_equiv
        elif key=="ocw":
            eval_f = is_equiv_ocw
        else:
            raise ValueError(f"key: {key} is not supported")
        
        complete_row_ = partial(complete_row, eval_f=eval_f, key=key)
        records = pqdm(records, complete_row_, n_jobs=8)

        # for row in tqdm(records, desc = key):
        #     for row in records:
        #         try:
        #             row["eval"] = eval_f(row["answer"], row["answer"])
        #             if key == "ocw": # if ocw, try another eval method 
        #                 row["eval_new"] = eval_f(row["answer"], row["answer"], approach_w_symexp=True)
        #         except Exception as e:
        #             row["eval"] = f"EVAL_FAIL! {str(e)}"

        coi = ["level", "type", "answer", "artificial_wrong", "eval"] if key == "math" else ["answer", "artificial_wrong", "eval", "eval_new"]
        # coi = ["level", "type", "answer", "eval"] if key == "math" else ["answer", "eval", "eval_new"]
        df = pd.DataFrame(records).loc[:, coi]
        
        if key == "math":
            groupwise = df.groupby("type").apply(lambda g: (g["eval"].apply(isfail).sum(), len(g)))
            print(groupwise)
        if key == "ocw":
            failold = df["eval"].apply(isfail).sum()
            failnew = df.eval_new.apply(isfail).sum()
            print("old", failold, "new", failnew)   

        newf = f"{key}_eval_answer2answer.jsonl"
        with jsl.open(newf, "w") as writer:
            writer.write_all(df.to_dict(orient="records"))
            print(newf)

