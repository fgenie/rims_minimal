import re
from typing import Dict, List, Union

import func_timeout

# to avoid matplotlib error by display in code execution
import matplotlib

matplotlib.use("Agg")

######################
## code execution ##
######################


def get_func_name_from_string(codestring: str) -> str:
    match = re.search(
        r"def (\w+)\(\):", codestring
    )  # kwargs만으로 정의되며 default가 포함된 함수는 처리하지 않음. 아쉬운 부분이지만 문제는 없음.
    if match:
        funcname = match.group(1)
        return funcname
    else:
        return None


def _execute(code, code_return: str):
    # these imports are for locals() (to provide `global() context` to exec()
    import itertools
    import math
    import random
    from fractions import Fraction

    import sympy
    import sympy as sp
    from sympy import Symbol
    from sympy import isprime as is_prime
    from sympy import symbols

    # pip installed olympiad, and marker to avoid frequent errors of math solving

    def _mock_input(prompt=""):
        """
        to ignore input() to be attempting user in the code
        """
        return ""

    try:
        locals_ = locals()
        locals_["input"] = _mock_input
        if (
            "import matplotlib" in code
            or "import matplotlib.pyplot" in code
            or "plt.figure" in code
        ):
            code = "import matplotlib\nmatplotlib.use('Agg')\n" + code

        exec(code, locals_)  # code로 local_ 딕셔너리 업데이트
        solution = locals_.get("solution", None)
        funcname = get_func_name_from_string(code)  # for nontrivial function names

        if solution is not None:
            ans = solution()  # this will use locals()
        elif funcname:  # if any function name appears
            new_code = "import math\n" + code + f"\nresult = {funcname}()"
            loc = {}
            locals__ = locals()
            locals__["input"] = _mock_input
            exec(new_code, locals__, loc)  # this will also use locals()

            ans = loc["result"]
        else:
            executed_code = (
                "import math\n"
                + "import datetime\n"
                + "\n".join([xx[4:] for xx in code.strip().split("\n")[1:-1]])
            )
            exec(executed_code, {"input": _mock_input}, locals())
            locals_ = locals()
            ans = locals_.get(code_return, None)

        # check if ans is sympy object so it needs to be converted by `sp.latex()`
        if isinstance(ans, sp.Basic):
            try:
                ans = sp.latex(ans)
            except Exception as e:
                print(e)
                print(f"{ans=} cannot be `sp.latex()`'d")
        return ans

    except Exception as exp:
        print("Executing code error", exp)
        print(f"code:\n{code}")
        print(f"{code_return=}")
        print(f"{(solution is None)=}")
        print(f"{funcname=}")
        return None


### executing a code
def safe_execute_turbo(code_string: str):
    def _convert_to_float_if_possible(ans):
        try:
            return float(ans)
        except Exception:
            return ans

    def _convert_to_str_if_not_none_nor_float(ans):
        if ans is not None and not isinstance(ans, float):
            try:
                ans = str(ans)
            except Exception as e:
                print(e)
                pass
        return ans

    # === find code snippets between def solution(): and return ===
    try:
        code_list = code_string.strip().split("\n")

        new_code_list = []
        all_codes = []
        code_return = "ans"

        for i in range(len(code_list)):
            if code_list[i].startswith("import "):
                all_codes.append(code_list[i])
            if re.search(r"def (\w+)\(", code_list[i]) and code_list[i].startswith(
                "def "
            ):  # avoid including inner function definition
                new_code_list.append(code_list[i])
                for j in range(i + 1, len(code_list)):
                    if code_list[j].startswith("    "):
                        new_code_list.append(code_list[j])
                    if code_list[j].startswith(
                        "    return "
                    ):  # affirms outtermost return
                        code_return = code_list[j].split("return ")[1].strip()
                        break  # it could possibly miss the return if the function written with if-elif-else return at the end, which might be scarce.
                all_codes.append("\n".join(new_code_list))
                new_code_list = []

        if all_codes:
            new_code = "\n\n".join(
                all_codes
            )  # all_codes[-1] # if we parsed more than one function, we need to use them all.

            ans = func_timeout.func_timeout(
                3,
                _execute,
                args=(
                    new_code,
                    code_return,
                ),
            )
            ans = _convert_to_float_if_possible(ans)
            ans = _convert_to_str_if_not_none_nor_float(ans)
        else:
            ans = None
    except (func_timeout.FunctionTimedOut, IndexError, NameError, SyntaxError):
        ans = None

    return ans


######################
## cot execution ##
######################


def _find_the_last_numbers(txt: str) -> str:
    # Regex pattern to match numbers with optional commas as thousand separators
    # and optional scientific notation
    pattern = r"[+\-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:[eE][+\-]?\d+)?"
    matches = re.findall(pattern, txt)

    if matches:
        # Replace commas to handle the number correctly as a standard numeric format
        return matches[-1].replace(",", "")
    else:
        return None


def _find_the_last_latex_expression(txt: str) -> Union[str, List]:
    latex_pattern = (
        r"(?:\\\[|\\\(|\$).+?(?:\\\]|\\\)|\$)"  # r'\\(?:\[|\().*?\\(?:\]|\))'
    )
    matches = re.findall(latex_pattern, txt, re.DOTALL)
    if matches:
        found = matches[-1]
    else:
        found = []
    return found


def extract_ans_from_cot_MATHnOCW(solution: str) -> str:
    """
    parsing symbolic or any form answer-string of interest from CoT result
    this is for parsing answers from cot solution of MATH/ocw_courses prompt
    see the corresponding prompt at `rims_minimal/src/utils/ocw_MATH_prompts.yaml`
    """
    prefix1 = "Final answer:"
    prefix2 = "The final answer is"
    suffix = ". I hope it is correct."  # this does not appear frequently in response... but let us use it just in case.

    # assume the solution followed the few-shot format
    # 1. Answer strictly followed the format in the few-shot examples
    solution = solution.split(prefix1)[-1].strip()
    # 2. answer partly followed the format
    solution = solution.split(prefix2)[-1].strip()
    solution = solution.split(suffix)[0].strip()

    # parsed above might still have unnecessary natural languages
    # 3-1. try to find some math expressions that is in latex format
    found_latex = _find_the_last_latex_expression(solution)
    if found_latex:
        part_of_interest = found_latex
    else:  # 3-2. last resort: find the last numbers
        found_numbers = _find_the_last_numbers(solution)
        if found_numbers:
            part_of_interest = found_numbers
        else:  # preserve the minimal-processed-string as a parsed result
            part_of_interest = solution
    return part_of_interest


def extract_num_turbo(solution: str):
    """
    parsing (executing) cot result into a float
    see the prompt at `src/utils/math_prompt.py`
    This is for GSM prompt (from Automatic Model Selection Reasoning https://arxiv.org/pdf/2305.14333.pdf)
    """
    ans: str = solution.strip().replace("So the answer is ", "")
    prd: Union[str, None] = _find_the_last_numbers(ans)
    prd = float(prd.replace(",", "").rstrip(".")) if prd else prd

    return prd
