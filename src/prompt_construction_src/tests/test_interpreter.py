import jsonlines as jsl
from pqdm.processes import pqdm
from tqdm import tqdm
import pandas as pd 

from utils.llm_query_utils import safe_execute_turbo 

def count_difference(row:dict)->dict:
    code = row["solmap"]["pal"]
    new_exec = safe_execute_turbo(code)
    old_exec = row["ansmap"]["pal"]
    row["newexec"] = new_exec
    row["oldexec"] = old_exec
    return row 
     


if __name__ == "__main__":
    # gsmjslf = "/Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RESULTS_v1/gsm_0613long/ablation/chatgpt0613long_rims_gsm.jsonl"
    # records = list(jsl.open(gsmjslf))
    # 6 None's, nothing changed

    OCW_RESULT = "/Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RESULTS_v1/ocw_0613long/chatgpt0613long_model_selection3_ocw.jsonl"
    records = list(jsl.open(OCW_RESULT))
    # as expected, sympy results are convereted 
    # 82 over 272 diff.
    # 58 None's

    # solmap.pal execute == ansmap.pal count
    for row in tqdm(records):
        row = count_difference(row)
    
    df = pd.DataFrame(records)
    mask = df.newexec != df.oldexec


    print(df[mask].loc[:, ["newexec", "oldexec"]].dropna()) # only None 's are figured differently executed 
    diff = len(df[mask]) - len(df[mask].loc[:, ["newexec", "oldexec"]].dropna())

    print("how many got different?")
    print(mask.sum(), len(mask))
    print("none", diff)