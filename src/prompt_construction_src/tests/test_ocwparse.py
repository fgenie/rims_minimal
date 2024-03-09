from utils.math_util import ocw_parse, INVALID_ANSWER

import jsonlines as jsl 
from tqdm import tqdm
from pqdm.processes import pqdm
import pandas as pd 



# load ocw_courses data 
records = list(jsl.open("/Users/seonils/dev/rims_minimal/dataset/ocw/ocw_course.jsonl"))

def parse_old_and_new(row):
    unparsed = row["answer"]
    newparse = ocw_parse(unparsed, use_old=False)
    oldparse = ocw_parse(unparsed, use_old=True)
    row["normalize_symbolic_expression"] = newparse
    row["normalize_final_answer"] = oldparse
    return row

# records = pqdm(records, parse_old_and_new, n_jobs=4)
for row in tqdm(records):
    row = parse_old_and_new(row)
df = pd.DataFrame(records)
# print(df.columns)
newparse = df.normalize_symbolic_expression
oldparse = df.normalize_final_answer 


def count_failures(s:pd.Series)->int:
    return f"invalid: {(s==INVALID_ANSWER).sum()}, symfail: {s.apply(lambda txt: str(txt) == 'i*(n*(v*(a*(l*(i*(da*(n*(s*(w*(e*r))))))))))').sum()}, exceptions: {s.apply(lambda txt: 'PARSE_FAIL!' in str(txt)).sum()}"

print("new", count_failures(newparse))
print("old", count_failures(oldparse))

df_ = df.loc[:, ["answer", "normalize_symbolic_expression", "normalize_final_answer"]]
df_.normalize_symbolic_expression = df_.normalize_symbolic_expression.apply(str)
df_.normalize_final_answer = df_.normalize_final_answer.apply(str)

with jsl.open("ocw_parsed_diff.jsonl", "w") as wd, jsl.open("ocw_parsed_eq.jsonl", "w") as we:
    eqmask = df_.normalize_symbolic_expression == df_.normalize_final_answer
    we.write_all(df_[eqmask].to_dict(orient="records"))
    wd.write_all(df_[~eqmask].to_dict(orient="records"))
    print("ocw_parsed_diff.jsonl written!")
    print("ocw_parsed_eq.jsonl written!")