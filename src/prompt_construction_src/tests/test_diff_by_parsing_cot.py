# %% [markdown]
# # OCW, MATH: Why low acc?
# > Possible flaws in
# > * equivalence function? (checked innocent)
# > * parsing? (**suspicious here.**)

# %%
import jsonlines as jsl
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
# from pqdm.processes import pqdm
from utils.math_util import math_check_answer



MATH_RESULT = "/Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RESULTS_v1/math_full_0613long/chatgpt0613long_model_selection3_math_merged.jsonl"
OCW_RESULT = "/Users/seonils/dev/rims_minimal/src/seonil_scripts/0_RESULTS_v1/ocw_0613long/chatgpt0613long_model_selection3_ocw.jsonl"

# %%
math_df = pd.DataFrame(jsl.open(MATH_RESULT))
ocw_df = pd.DataFrame(jsl.open(OCW_RESULT))

# # %%
# math_df.shape, ocw_df.shape
# math_df.columns

# %%
from utils.llm_query_utils import extract_ans_from_cot_MATHnOCW
from utils.math_util import is_equiv, is_equiv_ocw, normalize_final_answer

def diff_oldparse_newparse(df:pd.DataFrame):
    df["newpred_cot"] = df.solmap.apply(lambda d: extract_ans_from_cot_MATHnOCW(d["cot"]))
    df["oldpred_cot"] = df.ansmap.apply(lambda d: d["cot"])
    df["cot_changed"] = df.newpred_cot != df.oldpred_cot
    return df 

# %%
math_df_ = diff_oldparse_newparse(math_df)
ocw_df_ = diff_oldparse_newparse(ocw_df)


# math_df_.cot_changed.sum(), len(math_df_), ocw_df_.cot_changed.sum(), len(ocw_df_)

# %% [markdown]
# ## oldparse vs newparse of CoT solutions
# - I guess the parsing is done as our wish. 
# - But the solution did not (fewshot-inferenced by gsm fewshots, thus old parsing function will work better on it)

# %%
# math_df_.loc[:, ["newpred_cot", "oldpred_cot", "answer"]]

# %%
# ocw_df_.loc[:, ["newpred_cot", "oldpred_cot"]]


# %%
def before_and_after_accuracy(df, equiv_f:callable=None):
    newparse_correct = df.progress_apply(lambda row: equiv_f(row.newpred_cot, row.answer), axis="columns")
    oldparse_correct = df.progress_apply(lambda row: equiv_f(row.oldpred_cot, row.answer), axis="columns")
    results = {
        "new_acc": round(newparse_correct.mean(),3),
        "old_acc": round(oldparse_correct.mean(),3),
        "newparse_correct": (newparse_correct.sum(), len(newparse_correct)),
        "oldparse_correct": (oldparse_correct.sum(), len(oldparse_correct))
    } 
    return results
math_eq_nonorm = lambda x,y: is_equiv(str(x), str(y))
math_eq = lambda x,y: is_equiv(normalize_final_answer(str(x)), normalize_final_answer(str(y))) # == math_check_answer
ocw_eq_nonorm = lambda x,y: is_equiv_ocw(str(x), str(y)) # == ocw_check_answer
ocw_eq_norm = lambda x,y: is_equiv_ocw(normalize_final_answer(str(x)), normalize_final_answer(str(y)))

# %%
# math_res_nonorm = before_and_after_accuracy(math_df_, math_eq_nonorm) 
# math_res = before_and_after_accuracy(math_df_, math_eq) 
math_res = before_and_after_accuracy(math_df_, math_check_answer) 
# math_res_ocweval = before_and_after_accuracy(math_df_, ocw_eq_nonorm)

# %%
# ocw_res_nonorm = before_and_after_accuracy(ocw_df_, ocw_eq_nonorm)
# ocw_res_norm = before_and_after_accuracy(ocw_df_, ocw_eq_norm)

# %%
# for res in "math_res, ocw_res_sym, ocw_res_norm".split():
# for res in "math_res_nonorm, math_res, ocw_res_nonorm, ocw_res_norm".split():
# for res in "math_res_ocweval".split():
for res in "math_res".split():
    print(res)
    print(eval(res))



"""
math_res, (select!)
{'new_acc': 0.274, 'old_acc': 0.247, 'newparse_correct': (1370, 4996), 'oldparse_correct': (1232, 4996)}
({'new_acc': 0.266, 'old_acc': 0.247, 'newparse_correct': (1329, 4996), 'oldparse_correct': (1232, 4996)},)
ocw_res_nonorm, (select!) 
({'new_acc': 0.195, 'old_acc': 0.099, 'newparse_correct': (53, 272), 'oldparse_correct': (27, 272)},)


ABANDONNED:

math_res_nonorm,
({'new_acc': 0.083, 'old_acc': 0.237, 'newparse_correct': (415, 4996), 'oldparse_correct': (1183, 4996)},)
ocw_res_norm
{'new_acc': 0.136, 'old_acc': 0.099, 'newparse_correct': (37, 272), 'oldparse_correct': (27, 272)}


NEW TRIAL
ocweq_norm: {'new_acc': 0.273, 'old_acc': 0.248, 'newparse_correct': (1366, 4996), 'oldparse_correct': (1238, 4996)}
ocweq_nonorm(stdrd): 
"""