from utils.llm_query_utils import (_find_the_last_latex_expression, 
                                   _find_the_last_numbers, 
                                   extract_num_turbo, 
                                   extract_ans_from_cot_MATHnOCW,
                                   )
from utils.math_util import gsm_check_answer, ocw_check_answer

import jsonlines as jsl
import pandas as pd 
from tqdm import tqdm
tqdm.pandas()

import regex


def old_extract_num_turbo(solution:str)->float:
    ans = solution.strip().split("\n")[-1].replace("So the answer is ", "")    
    prd = [x[0] for x in regex.finditer(r"[\d\.,]+", ans) if regex.search(r"\d", x[0])]
    if prd:
        try:
            num = prd[-1]
            num = num.replace(",", "")
            num = float(num)
        except:
            num = None
    else:
        num = None
    return num

def get_cot_ans_df(jslf)->pd.Series:
    df = pd.DataFrame(jsl.open(jslf))
    cot_sln = df.solmap.apply(lambda d: d["cot"] if "cot" in d else None)
    cot_parsed = df.ansmap.apply(lambda d: d["cot"] if "cot" in d else None)
    cot_ans = df.answer
    return pd.DataFrame({"cot_sln": cot_sln, "bugparse": cot_parsed, "cot_ans": cot_ans}).dropna()


# from run_evaluation_new.py
def eval_gsm_svamp(df, 
                   return_flag:bool=False, 
                   submission_col_already_exists:bool=False):
    if not submission_col_already_exists:
        df["submission"] = df.majority_ans
    df.submission = df.submission.astype("str")

    equiv_flag = df.progress_apply(
        lambda row: gsm_check_answer(row.submission, row.cot_ans), 
        axis=1
        )
    if return_flag:
        return equiv_flag
    else:
        return equiv_flag.sum() if len(df)>0 else 0

if __name__ == "__main__":
    gsm_file = "../../outputs/gsm8K_test_dt.gsm/chatgpt0613long/model_selection_prompts/03_23_14_35_13.jsonl"
    ocw_file = "../../outputs/ocw_course_dt.ocw/chatgpt0613long/model_selection_prompts/03_23_15_38_01.jsonl"

    # gsm
    df = get_cot_ans_df(gsm_file)
    # "bugparse"
    df["revised"] = df.cot_sln.apply(extract_num_turbo)
    df["oldermethod"] = df.cot_sln.apply(old_extract_num_turbo)
    df["_find_the_last_numbers"] = df.cot_sln.apply(_find_the_last_numbers)
    
    # check accuracy
    df['submission'] = df.bugparse
    bugparse_acc = eval_gsm_svamp(df, submission_col_already_exists=True)
    
    df['submission'] = df.revised
    revised_acc = eval_gsm_svamp(df, submission_col_already_exists=True)

    df['submission'] = df.oldermethod
    oldermethod_acc = eval_gsm_svamp(df, submission_col_already_exists=True)

    print(f"bugparse_acc: {bugparse_acc}")
    print(f"revised_acc: {revised_acc}")
    print(f"oldermethod_acc: {oldermethod_acc}")

    df = df.drop(columns = ['submission'])
    df.to_json("gsm_ans.jsonl", orient='records', lines=True)
    

    """
    bugparse_acc: 188
    revised_acc: 939
    oldermethod_acc: 983
    """

    df = get_cot_ans_df(ocw_file)
    df.rename(columns={"bugparse": "parsed"})
    df.to_json("ocw_ans.jsonl", orient='records', lines=True)