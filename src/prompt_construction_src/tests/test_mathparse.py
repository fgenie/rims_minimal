from utils.math_util import normalize_final_answer, INVALID_ANSWER, math_parse

import jsonlines as jsl 
from tqdm import tqdm
from pqdm.processes import pqdm
import pandas as pd 



# load ocw_courses data 
records = list(jsl.open("/Users/seonils/dev/rims_minimal/dataset/MATH/MATH-full.jsonl"))


# records = pqdm(records, parse_old_and_new, n_jobs=4)
for row in tqdm(records):
    try:
        row["newparse"] = math_parse(row["answer"])
    except Exception as e:
        row["newparse"] = f"PARSE_FAIL! {str(e)}"
df = pd.DataFrame(records)
# print(df.columns)
newparse = df.newparse
# oldparse = df.oldparse 


def count_failures(s:pd.Series)->int:
    return f"invalid: {(s==INVALID_ANSWER).sum()}, symfail: {s.apply(lambda txt: str(txt) == 'i*(n*(v*(a*(l*(i*(da*(n*(s*(w*(e*r))))))))))').sum()}, exceptions: {s.apply(lambda txt: 'PARSE_FAIL!' in str(txt)).sum()}"

print("new", count_failures(newparse))

df_ = df.loc[:, ["answer", "newparse"]] #, "oldparse"]]
df_.newparse = df_.newparse.apply(str)
# df_.oldparse = df_.oldparse.apply(str)

with jsl.open("math_parsed.jsonl", "w") as w:
    w.write_all(df_.to_dict(orient="records"))
    print("math_parsed.jsonl written!")