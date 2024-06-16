"""
raw query outputs
-> gradable / analyseable files
-> readable by rims/sg script so that only selection-needed rows get inferred


----
resources:
    src/run_inference.py
    src/utils/llm_query_utils.py # code execution, parsing, etc...


"""

import re
from typing import Dict, List, Literal, Union


#################
##### rims ######
#################
def parse_method2(methodstr: str) -> str:
    # works for --rimsprompt option
    normalized = methodstr.replace("-", " ").replace("_", " ").lower()
    norm2short = {
        "chain of thought": "cot",
        "cot": "cot",
        "program aided language modeling": "pal",
        "program aided language model": "pal",
        "pal": "pal",
        "plan and then code": "p2c",
        "p2c": "p2c",
    }  # this should be key as abb, and value as a set of component patterns for capturing
    for k in norm2short.keys():
        if k in normalized:
            return norm2short[k]
    else:
        return methodstr


# rims prompt: cot answer extracting postprocessing
def parse_num_from_answer(rawstr) -> float:
    """
    used for parsing number out from Answer (dec 4 exp)
    """
    rawstr = rawstr.replace(",", "")
    ptn = r"(-?\d+\.\d+|\d+)"
    nums = re.findall(ptn, rawstr)
    if not nums:
        return None
    else:  # more than one number
        return float(nums[-1])


#################
# simple greedy #
#################
def postprocess_selection(selection_str: str) -> str:
    ptn = r"\([A-C]\)"
    matches = re.findall(ptn, selection_str)
    if matches:
        choice = matches[0]

        choice2method = {"(A)": "cot", "(B)": "pal", "(C)": "p2c"}

        return choice2method[choice]
    else:
        return None


#####################
# individual method #
#####################
def postprocess_plan(rawanswer: str):
    # lines = [l for l in rawanswer.split('\n') if '</end>' not in l]
    lines = rawanswer.split("\n")
    if len(lines) >= 1:
        plan_ = "\n".join(lines)
    else:
        print("plan gen failed")
        print(f"{rawanswer=}")
        plan_ = ""
    return plan_


### python code parsing for p2c... I know it's non-equivalent to the other function here. Used only for p2c
def postprocess_code(rawanswer: str):
    # 1 removing starting wrap ```
    if "```python" in rawanswer:
        rawanswer = rawanswer.split("```python")[-1]
    elif rawanswer.startswith("```"):
        rawanswer = rawanswer.split("```")[-1]

    # 2 removing ``` at the end
    code = rawanswer.split("```")[0]  # ending ``` removal

    # 3 remove prints
    code = code.replace("print(", "# print(")
    code.strip()

    return code
