"""
raw query outputs
-> gradable / analyseable files
-> readable by rims/sg script so that only selection-needed rows get inferred


----
resources:
    src/run_inference.py
    src/utils/llm_query_utils.py # code execution, parsing, etc...


"""


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


#####################
### majority vote ###
#####################


def bucket_count_floating_numbers(numbers: List, tolerance: float = 1e-3) -> Counter:
    """
    used for `get_concordant_answer_n`

    # Example usage:
    numbers = [0.001, 0.002, 0.0025, 0.003, 1.000, 1.0005, 0.999]
    tolerance = 1e-3
    result = bucket_count_floating_numbers(numbers, tolerance)
    print(result)
    """

    # Function to bucket the numbers
    def bucket_number(num, tolerance):
        # 0.00251 will assign to 0.003, 0.0025 will assign to 0.002
        try:
            buck_num = round(num / tolerance) * tolerance
        except Exception as e:
            print(e)
            print("`bucket_count_floating_numbers()` fail!")
            buck_num = None
        return buck_num

    # Create a dictionary to count occurrences of each bucketed value
    bucket_counts = {}

    for number in numbers:
        # Bucket the number
        bucketed_number = bucket_number(number, tolerance)

        # Update the count for this bucket
        if bucketed_number is None:
            continue
        elif bucketed_number in bucket_counts:
            bucket_counts[bucketed_number] += 1
        else:
            bucket_counts[bucketed_number] = 1

    # Return the dictionary with counts of each bucketed value
    return Counter(bucket_counts)


def bucket_count_ocw_math_ans(answers: List[str], dataset_type: Literal["ocw", "math"]):
    """
    used for `get_concordant_answer_n`
    """
    answers_counts = Counter(answers)
    if len(answers_counts) == 1:
        pass
    else:
        eq_f = (
            math_util.math_check_answer
            if dataset_type == "math"
            else math_util.ocw_check_answer
        )
        for a1, a2 in combinations(answers_counts.keys(), 2):
            if eq_f(a1, a2):
                sumcount = answers_counts[a1] + answers_counts[a2]
                answers_counts[a1] = sumcount
                answers_counts[a2] = sumcount

    return answers_counts


def get_concordant_answer_n(
    answers: List, dataset_type: Literal["gsm", "svamp", "ocw", "math"] = "gsm"
) -> Union[float, str, List, None]:
    """
    n>3 version of the get_concordant_answer
    """
    answers_no_none = [a for a in answers if (a is not None and a != "None")]
    if not answers_no_none:
        return None

    if dataset_type in ["svamp", "gsm"]:
        # relatively, arithmetics has less need for tolerance comparison. So, we can use Counter first.
        majority, count = Counter(answers_no_none).most_common(1)[0]
        if count >= 2:
            return majority if isinstance(majority, float) else None
        # no majority: count = 0, 1
        else:
            # continue to check if 1e-3 tolerance same thing exist
            if len(answers_no_none) == 0:
                return None
            elif len(answers_no_none) == 1:
                majority = answers_no_none.pop()
                return majority if isinstance(majority, float) else None
            else:  # len(answers_no_none) >= 2:
                # count the floating numbers
                bucket_counts = bucket_count_floating_numbers(answers_no_none)
                # if there is a bucket with more than 1 number, return the bucket
                majority_, count_ = bucket_counts.most_common(1)[0]
                if count_ > 1:
                    return majority_
                else:
                    return None

    else:  # dataset_type in ["math", "ocw"]:
        if len(answers_no_none) == 0:
            return None
        elif len(answers_no_none) == 1:
            return answers_no_none.pop()
        else:  # len(answers_no_none) >= 2:
            # count the floating numbers
            bucket_counts = bucket_count_ocw_math_ans(
                answers_no_none, dataset_type=dataset_type
            )
            # if there is a bucket with more than 1 number, return the bucket
            majority_, count_ = bucket_counts.most_common(1)[-1]
            maj2_, count2_ = bucket_counts.most_common(2)[-1]

            majorities = [majority_, maj2_] if count_ == count2_ else [majority_]
            if count_ > 1:
                return majorities
            else:
                return None


def get_concordant_answer(
    answers: list,
    ensure_unanimity: bool = False,
    dataset_type: Literal["gsm", "svamp", "ocw", "math"] = "gsm",
):
    """
    check if there is a pair of concordant answers.
    input: cot_ans, pal_ans, p2c_ans, [, ...]
    output: ans if concordant else None

    *recommend to put answers in the order of cot going first (usually they are intgers)

    **for math and ocw, safe_execute_turbo polishes returned answer with `sp.latex` if the result is sympy object (See the last line of `def _execute`)
    """
    if ensure_unanimity:
        if len(set(answers_no_none)) == 1:
            majority = answers_no_none.pop()
            return majority if isinstance(majority, float) else None
        else:
            return None

    answers_no_none = [a for a in answers if (a is not None and a != "None")]

    if dataset_type in ["svamp", "gsm"]:
        majority, count = Counter(answers_no_none).most_common(1)[0]
        if count >= 2:
            return majority if isinstance(majority, float) else None
        else:  # count = 1
            # continue to check if 1e-3 tolerance same thing exist
            if len(answers_no_none) == 0:
                return None
            elif len(answers_no_none) == 1:
                majority = answers_no_none.pop()
                return majority if isinstance(majority, float) else None
            elif len(answers_no_none) == 2:
                try:
                    if abs(answers_no_none[0] - answers_no_none[1]) < 1e-3:
                        return (
                            answers_no_none[0]
                            if isinstance(answers_no_none[0], float)
                            else None
                        )
                    else:
                        return None
                except:
                    return None
            else:  # >=3
                for a1, a2 in combinations(answers_no_none, 2):
                    try:
                        if abs(a1 - a2) < 1e-3:
                            return a1
                        else:
                            return None
                    except:
                        continue
                return None  # no concordant answers
    elif dataset_type in ["math"]:
        answers_normalized = [
            math_util.normalize_final_answer(str(a)) for a in answers_no_none
        ]
        if ensure_unanimity:
            raise NotImplementedError("ensure_unanimity is not supported for now: math")
        else:
            if len(answers_normalized) == 0:
                return None
            elif len(answers_normalized) == 1:
                return answers_no_none.pop()
            elif len(answers_normalized) == 2:
                try:
                    if math_util.is_equiv(answers_normalized[0], answers_normalized[1]):
                        return answers_no_none[0]
                except Exception as e:
                    print(e)
                return None
            else:  # len()==3
                revert_normalized = dict(zip(answers_normalized, answers_no_none))
                for a1, a2 in combinations(answers_normalized, 2):
                    try:
                        if math_util.is_equiv(a1, a2):
                            return revert_normalized[a1]
                    except Exception as e:
                        print(e)
                        continue
                return None  # no concordant answers
    elif dataset_type in ["ocw"]:
        if ensure_unanimity:
            raise NotImplementedError("ensure_unanimity is not supported for now: ocw")
        else:
            if len(answers_no_none) == 0:
                return None
            elif len(answers_no_none) == 1:
                return answers_no_none.pop()
            elif len(answers_no_none) == 2:
                try:
                    if math_util.is_equiv_ocw(answers_no_none[0], answers_no_none[1]):
                        return answers_no_none[0]
                    else:
                        return None
                except Exception as e:
                    print(e)
                return None
            else:  # len()==3
                for a1, a2 in combinations(answers_no_none, 2):
                    try:
                        if math_util.is_equiv_ocw(a1, a2):
                            return a1
                    except Exception as e:
                        print(e)
                        continue
                return None  # no concordant answers
