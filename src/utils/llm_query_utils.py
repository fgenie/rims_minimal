import math
import os
import re
from collections import Counter  # , defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Literal, Union

import func_timeout
import jsonlines as jsl
import yaml
from openai import AzureOpenAI, OpenAI
from sympy.parsing.latex import parse_latex

from . import math_prompt, math_util
from .cost_tracking import CountTokens

# Get the absolute path of the current script
THIS_PARENT = Path(__file__).parent.resolve()

# Construct the path to the openai_key.txt file


# vllm/openai server that serves chatmodel
client = OpenAI(
    base_url="http://172.21.0.1:8000/v1",
    api_key="no_need",
    timeout=120,
    max_retries=4,
)


def merge_system_into_first_user(msgs) -> list:
    if msgs[0]["role"] == "user":
        msgs_new = msgs
    else:  # system role
        system_turn = msgs[0]
        start_turn = msgs[1]
        msgs[1]["content"] = system_turn["content"] + "\n\n" + start_turn["content"]
        msgs_new = msgs[1:]
    return msgs_new


# @rate_limiter.is_limited()
def query_with_openlimit(**chat_params):
    return client.chat.completions.create(**chat_params)


def exception_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(e)
            return None

    return wrapper


### almost same to string.Template, but with custom delimiter ( [QUESTION] == ${QUESTION}, to avoid `$` used frequently in price-related questions )
class PromptStr(str):
    def __init__(self, template_str):
        super().__init__()
        self += template_str

    def sub(self, placeholder_name: str, tobe: str):
        return PromptStr(self.replace(f"[{placeholder_name}]", str(tobe)))

    def sub_map(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.get_placeholder_names():
                self = self.sub(k, v)
        return self

    def get_placeholder_names(self) -> list:
        return re.findall(r"\[(.*?)\]", self)


### llm query functions ###
# @CountTokens
def query_cot(
    question: str,
    dataset_type: Literal["gsm", "ocw", "math"] = "",
    temperature: float = 0.0,
    backbone: str = "chatgpt",
    n: int = 1,
    seed: int = 777,
    max_tokens: int = 2048,
):
    """
    This function is used to query OpenAI for CoT solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        temperature: the temperature used in CoT
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the CoT solution
    """

    query_message = get_cot_prompt(
        question, backbone=backbone, dataset_type=dataset_type
    )
    # print(query_message)
    model_name = backbone2model(backbone)

    completions = []
    # if backbone2model(backbone) == backbone: # openllm (assumed to have no sys)
    #     query_message = merge_system_into_first_user(query_message)
    resp = query_with_openlimit(
        model=model_name,
        max_tokens=max_tokens,
        stop=["\n\n\n", "<|end|>"],
        messages=query_message,
        temperature=temperature,
        # top_p=1.0,
        seed=seed,
        n=n,
    )

    completions = [choice.message.content for choice in resp.choices]
    return completions, query_message, resp


# actual llm query function for p2c method
# @CountTokens
def _query(  # key,
    model_name: str = "gpt-3.5-turbo-1106",  # "gpt-3.5-turbo-16k-0613",
    max_tokens: int = 1024,
    stop: str = None,
    messages=None,
    temperature=0.0,
    # top_p=1.0,
    n=1,
    mode="plan",
    seed=777,
):  # mode = plan or code
    """
    atomic query func for query_plancode
    """
    try:
        resp = query_with_openlimit(
            model=model_name,
            max_tokens=max_tokens,
            stop=stop,
            messages=messages,
            temperature=temperature,
            # top_p=# top_p,
            n=n,
            seed=seed,
            # timeout=120,
        )
        # completion_usage = resp.usage
        if n == 1:
            content = resp.choices[0].message.content  # str
            if mode == "plan":
                plan = postprocess_plan(content)  # it will complain when failing
                return plan, resp  # completion_usage
            elif mode == "code":
                code = postprocess_code(content)
                return code, resp  # completion_usage
        else:  # n>1
            contents = [ch.message.content for ch in resp.choices]
            postprocess = postprocess_plan if mode == "plan" else postprocess_code
            res_strs = [postprocess(c) for c in contents]
            return res_strs, resp  # completion_usage
    except Exception as e:
        print(e)
        return None, str(e)


# p2c: querying plan and code separately inside
def query_plancode(
    question: str,  # data: dict,
    plan_temperature: float = 0.0,
    code_temperature: float = 0.0,
    backbone: str = "chatgpt1106",  # "gpt-3.5-turbo-16k-0613",
    n=1,
    seed: int = 777,
    dataset_type: str = "gsm",  # ocw math
    # max_tokens: int = 1024,
):
    """
    PAL variant:
    1. generate planning for the given question
    2. based on 1, generate code like PAL does.

    args:
        mostly same arguments with `query_pal()` below
    returns:
        [list of codes], [list of plans (1)], {codequery: str, planquery: str}
    """
    # specify model
    model_name = backbone2model(backbone)

    k_fewshot = 8  # default init 8, for openLLM cases
    if backbone.startswith("gpt4"):
        # print(f'gpt-4 uses k_fewshot=5 as default (p2c fs_prompting)')
        k_fewshot = 5
    elif backbone.startswith("chatgpt"):  # ("gpt-3.5-turbo-16k-0613"):
        # print(f'gpt-3.5 uses k_fewshot=8 as default (p2c fs-prompting)')
        k_fewshot = 8

    # generate plan (retry included)
    plan_query_msg = get_plan_prompt(question, k_fewshot=k_fewshot)
    # print(plan_query_msg)

    plan_max_tokens_d = {
        "gsm": 1024,
        "ocw": 1024,
        "math": 1024,
    }
    code_max_tokens_d = {
        "gsm": 2048,
        "ocw": 2048,
        "math": 2048,
    }

    # if backbone2model(backbone) == backbone: # openllm (assumed to have no sys)
    #     plan_query_msg = merge_system_into_first_user(plan_query_msg)
    plan, _ = _query(
        model_name=model_name,
        max_tokens=plan_max_tokens_d[dataset_type],
        stop=["Question: ", "<|end|>"],
        messages=plan_query_msg,
        temperature=plan_temperature,
        # top_p=1.0,
        # n=n,
        n=1,  # plan*1 + code*n (bad for p2c acc but perf. consideration)
        mode="plan",
        seed=seed,
    )

    # if n == 1:
    if plan:
        code_query_msg = get_plan2code_prompt(question, plan=plan, k_fewshot=k_fewshot)
        # print(code_query_msg)
        # if backbone2model(backbone) == backbone: # openllm (assumed to have no sys)
        #     code_query_msg = merge_system_into_first_user(code_query_msg)
        code, _ = _query(
            model_name=model_name,
            max_tokens=code_max_tokens_d[dataset_type],
            stop=["Question: ", "<|end|>"],
            messages=code_query_msg,
            temperature=code_temperature,
            # top_p=1.0,
            n=n,
            mode="code",
            seed=seed,
        )  # ,
        if not code:
            return (
                [None],
                [plan],
                {"codequery": code_query_msg, "planquery": plan_query_msg},
            )
        else:
            return (
                [code] if n == 1 else code,  # n>0 --> List[str]
                [plan],
                {"codequery": code_query_msg, "planquery": plan_query_msg},
            )
    else:
        return None, None, {"codequery": None, "planquery": plan_query_msg}
    # else:  # n>1
    #     if plan:
    #         plans = plan
    #         code_query_msgs = [
    #             get_plan2code_prompt(question, plan=p, k_fewshot=k_fewshot)
    #             for p in plans
    #         ]
    #         codes = [
    #             _query(
    #                 model_name=model_name,
    #                 max_tokens=1024,
    #                 stop="Question: ",
    #                 messages=cqm,
    #                 temperature=code_temperature,
    #                 # top_p=1.0,
    #                 n=n,
    #                 mode="code",
    #                 seed=seed,
    #             )[0]
    #             for cqm in code_query_msgs
    #         ]  # it is O(n) times slow... OMG...

    #         if not codes:
    #             return (
    #                 [None] * n,
    #                 plans,
    #                 {"codequery": code_query_msgs, "planquery": plan_query_msg},
    #             )
    #         else:
    #             return (
    #                 codes,
    #                 plans,
    #                 {"codequery": code_query_msgs, "planquery": plan_query_msg},
    #             )
    #     else:
    #         return None, None, {"codequery": None, "planquery": plan_query_msg}


# @CountTokens
def query_pal(
    question: str,
    temperature: float,
    backbone: str,
    n=1,
    seed=777,
    dataset_type: Literal["gsm", "ocw", "math", "svamp"] = None,
    max_tokens: int = 1024,
):
    """
    This function is used to query OpenAI for PAL solutions.

    Args:
        data: a dict containing the question and answer
        key: the OpenAI API key
        temperature: the temperature used in PAL
        backbone: ChatGPT or GPT-4

    Returns:
        completions: a list containing the PAL solution
    """
    if dataset_type not in "gsm ocw math svamp":
        raise ValueError(f"query_pal(): {dataset_type=} is not supported")

    query_message = get_pal_prompt(
        question, backbone=backbone, dataset_type=dataset_type
    )
    # print(query_message)
    model_name = backbone2model(backbone)
    # if backbone2model(backbone) == backbone: # openllm (assumed to have no sys)
    #     query_message = merge_system_into_first_user(query_message)
    completions = []
    resp = query_with_openlimit(
        model=model_name,
        max_tokens=max_tokens,
        stop=["\n\n\n", "<|end|>"],
        messages=query_message,
        temperature=temperature,
        # top_p=1.0,
        seed=seed,
        n=n,
    )

    completions = [choice.message.content for choice in resp.choices]
    return completions, query_message, resp


# @CountTokens
def query_selection(
    question: str,
    backbone: str,
    dataset_type: Literal["gsm", "ocw", "math"] = None,
    cot_solution: str = "",
    pal_solution: str = "",
    p2c_plan_code_solution: str = "",  # former, this was actually List[str] and get_select_prompt() did pop()'d p2c solution out of list... which is super ugly and confusing
    max_tokens: int = 512,
    temperature: float = 0.0,  # will not be modified
    n: int = 1,  # will not be used
):
    if n != 1:
        raise ValueError("query_selection(): n need to be 1")
    if temperature != 0.0:
        raise ValueError("query_selection(): temperature need to be 0.0")

    def postprocess_selection(selection_str: str) -> str:
        ptn = r"\([A-C]\)"
        matches = re.findall(ptn, selection_str)
        if matches:
            choice = matches[0]

            choice2method = {"(A)": "cot", "(B)": "pal", "(C)": "p2c"}

            return choice2method[choice]
        else:
            return None

    model_name = backbone2model(backbone)

    cot_pal_p2c_solution_list = [cot_solution, pal_solution, p2c_plan_code_solution]
    cot_pal_p2c_solution_list = [
        s for s in cot_pal_p2c_solution_list if s
    ]  # remove p2c if empty

    cot_pal_p2c_solution_d = dict(zip("cot pal p2c".split(), cot_pal_p2c_solution_list))

    selection_message = get_select_prompt2(
        question,
        cot_pal_p2c_solution_d,
        dataset_type=dataset_type,
    )

    # if backbone2model(backbone) == backbone: # openllm (assumed to have no sys)
    #     selection_message = merge_system_into_first_user(selection_message)

    response = query_with_openlimit(
        model=model_name,
        max_tokens=max_tokens,
        seed=777,  # added on dec 21
        stop=["\n\n", "<|end|>"],
        messages=selection_message,
        temperature=temperature,
        # top_p=1.0,
        n=n,
    )
    # completion_usage = response.usage
    select_str = response.choices[0].message.content

    final_answer = postprocess_selection(select_str)
    return final_answer, select_str, response  # completion_usage  # 'pal'|'p2c'|'cot'


# @CountTokens
def query_rims_inference(
    question: str,
    prompt_f: str,
    backbone: str,
    temperature: float = 0.0,
    n: int = 1,
    max_tokens: int = 2048,  # (rims prompts w/o question is ~ 2400 tokens with 3 blurbs + system)
    # continue_writing_gpt_messages: list = None,
    stop_tok=None,
    dataset_type: str = None,
) -> tuple:
    model_name = backbone2model(backbone)

    # def convert_to_turns(prompt:str, q:str='') -> list:
    #     assert q, f"question should be given {q=}"
    #     chunks = prompt.split("\n"*4)
    #     __origsys, *blurbs, qoi = chunks

    #     def _blurb2usr_asst(blurbstr:str)->list:
    #         idx = blurbstr.find("`Method`") # find from the left
    #         user = blurbstr[:idx].strip()
    #         asst = blurbstr[idx:].strip()
    #         usr_asst_messages = [
    #             {"role": "user", "content": user},
    #             {"role": "assistant", "content": asst}
    #         ]
    #         return usr_asst_messages

    #     def _qoi2questiononly(blurbstr:str)->list:
    #         idx = blurbstr.rfind("`Question`") # find from the left
    #         others = blurbstr[:idx].strip()
    #         q = blurbstr[idx:].strip()
    #         return others, q
    #     #SYS4TURN is actually, a recombination of part of instruction and original system message part
    #     SYS4TURN = \
    #     """You are now solving math word problems. You brilliantly detects the errors in the wrong solution and find `Workaround Method` to correct the solution. The methods you are taking are as follows. Each has its strength and weakness:

    # - Chain of Thought (cot): Solving problem with writing steps of reasoning in a natural language. Might help correct understanding of the problem but this could be weaker in precise computation.
    # - Program-aided Language Modeling (pal): Using python language to reason and obtain an accurate answer to the given question, but this could be weaker in understanding the problem.
    # - Plan-and-then-Code (p2c): When a question seems requiring amount of steps to reach the answer, write plans first for what to compute and write a python code to it for solving the problem. However if planning goes wrong, the code will also be wrong. If any steps of planning provided before programming, then it will be considered as Plan-and-then-Code.

    # Try the question with the choice of your `Method`, and evaluate the `Answer`. If your `Attempt` is considered wrong, identify the `Mistakes` and reason to take `Workaround Method` by writing `Hint for a better Method choice`. Based on it, make a correct reattempt."""

    #     blurbs = [_blurb2usr_asst(b.strip()) for b in blurbs]
    #     ___, qoi = _qoi2questiononly(qoi.replace("[QUESTION]", q))
    #     messages = [
    #         {"role": "system", "content": SYS4TURN},
    #         *chain(*blurbs),
    #         {"role": "user", "content": qoi}
    #     ]
    #     return messages

    def parse_raw_modif(rawqueryout: str) -> dict:
        """
        helper for Attempt 1,2,3... variants

        1/ read prompt to detect what to parse (`Some string here` <-- to be parsed)
        2/ and then parse those into a dict
        """
        # read the output and get what to parse
        pattern = r"`(.*?)`:"
        to_parse = re.findall(pattern, rawqueryout)
        to_parse = list(set(to_parse) - {"Evaluation"})

        # read the output again to parse the designated fields
        parse_dd = dict()

        duplicated = 1

        for fld in to_parse:
            # pattern = rf"`{fld}`:\s*(.*?)(?=`|$)"
            # pattern = rf"`{fld}`:\s*(?:```)?(.*?)(?:```)?(?=`|$)"
            pattern = rf"`{fld}`:\s*(?:```)?([\s\S]*?)(?=(?:```)?\n`[A-Z]|$)"
            matches = re.findall(pattern, rawqueryout, re.DOTALL)
            if fld in {
                "Mistakes",
                "Hint for a better Method choice",
                "Workaround Method",
                "Method",
            }:  # Method supposed not to appear multiple times, for chatgpt, it happens, and maybe for other plms too.
                parse_dd[fld] = matches[::duplicated]
            else:
                duplicated = max(duplicated, len(matches))
                if len(matches) > 0:
                    parse_dd[fld] = matches[0].strip()
                else:
                    parse_dd[fld] = ""

        for (
            k
        ) in (
            parse_dd.keys()
        ):  # found erratic parsings of the rims code solutions (``` at the end not removed properly)
            if k.startswith("Attempt "):
                if parse_dd[k].strip().endswith("```"):
                    parse_dd[k] = parse_dd[k].strip().rstrip("```").strip()
                if parse_dd[k].startswith("python\n"):
                    parse_dd[k] = parse_dd[k].replace("python\n", "").strip()

        if "Method" not in parse_dd.keys():
            parse_dd["Method"] = []  # for later pop error prevention

        return parse_dd

    def process_rims_out_dict(parse_dd: dict, dataset_type=None) -> dict:
        """
        in:
            parsed_dict: contains fields that is directly related to the prompt response such as...
                Attempt 1: solution1 string
                Answer 1: solution1 answer (raw string)
                Mistakes: [solution 1,2,3,...'s mistake string]
                ...
        out:
            eval_friendly_d (dict): contains eval-friendly parsed fields
                good_solution: solution string at the last
                good_ans: correct answer executed above
                good_method: correct method abbreviation (e.g. cot)
                bad_ans: [list of wrong answers]
                bad_method: [list of wrong methods before the correct one]
                bad_solutions: [list of wrong solutions before the correct one]
                mistakes: [list of mistakes]
                hint: [list of hints]

        """

        def get_answer_rims(
            solution: str, ans: str = None, method: str = "", dataset_type: str = "gsm"
        ):
            pred = ans
            try:
                if method == "cot":  # if method is given, follow it.
                    pred = (
                        parse_num_from_answer(ans)
                        if ans is not None and dataset_type == "gsm"
                        else ans
                    )
                elif method in ["pal", "p2c"]:
                    pred = safe_execute_turbo(solution)
            except Exception as e:
                try:
                    pred = safe_execute_turbo(solution)
                except:
                    pred = (
                        parse_num_from_answer(ans)
                        if ans is not None and dataset_type == "gsm"
                        else ans
                    )
                finally:
                    print(
                        f"Exception at llm_query_utils:query_rims_inference:process_rims_out_dict:get_answer_rims() --> using `ans` for get_anser_rims w/o executing\n\noriginal exception message:\n{e}"
                    )
            return pred

        attempts_keys = sorted([k for k in parse_dd.keys() if "Attempt" in k])
        ans_keys = sorted([k for k in parse_dd.keys() if "Answer" in k])
        # method_keys = sorted([k for k in parse_dd.keys() if 'Method' in k])

        if (
            ans_keys and attempts_keys
        ):  # answer and solutions inside. additionally Method key is also in the parse_dd
            good_solution = parse_dd[attempts_keys[-1]] if attempts_keys else None

            # reflection count
            did_reflect = 0
            if "Mistakes" in parse_dd.keys():
                did_reflect += len(parse_dd["Mistakes"])

            if "Workaround Method" in parse_dd.keys():
                did_reflect += len(parse_dd["Workaround Method"])
                good_method = parse_method2(parse_dd["Workaround Method"][-1])
                bad_method = [parse_method2(parse_dd["Method"].pop())]
                if len(parse_dd["Workaround Method"]) > 1:
                    bad_method += [
                        parse_method2(mstr)
                        for mstr in parse_dd["Workaround Method"][:-1]
                    ]

                # ans and solutions
                good_ans = parse_dd[ans_keys[-1]]
                bad_ans = [parse_dd[ak] for ak in ans_keys[:-1]]

                good_solution = parse_dd[attempts_keys[-1]]
                bad_solution = [parse_dd[atk] for atk in attempts_keys[:-1]]

            elif "Method" in parse_dd.keys():  # no reflection (solved at once)
                good_method = parse_method2(parse_dd["Method"][-1])
                bad_method = [parse_method2(m) for m in parse_dd["Method"][:-1]]

                # ans and solutions
                good_ans = parse_dd[ans_keys[-1]]
                bad_ans = [parse_dd[ak] for ak in ans_keys[:-1]]

                good_solution = parse_dd[attempts_keys[-1]]
                bad_solution = [parse_dd[atk] for atk in attempts_keys[:-1]]

            else:  # no "Method" key
                good_method = ""
                bad_method = []

                good_ans = parse_dd[ans_keys[-1]]
                bad_ans = []

                good_solution = parse_dd[attempts_keys[-1]]
                bad_solution = []

        else:  # rims queried for evaluation only. no answer nor solutions.
            did_reflect = 0
            good_solution = None
            good_method = None
            good_ans = None
            bad_solution = []
            bad_ans = []
            bad_method = []

        mistakes = []
        hint = []
        if "Mistakes" in parse_dd.keys():
            mistakes = parse_dd["Mistakes"]
        if "Hint for a better Method choice" in parse_dd.keys():
            hint = parse_dd["Hint for a better Method choice"]

        if not len(bad_solution) == len(bad_ans) == len(bad_method):
            print(f"{bad_solution=}", f"{bad_ans=}", f"{bad_method=}")
            print(f"{good_solution=}", f"{good_ans=}", f"{good_method=}")
            print(f"{bad_solution=} possibly repetition generated (chatgpt, temp 0)")
            # raise ValueError(
            #     f"{bad_solution=} possibly repetition generated (chatgpt, temp 0)"
            # )  # the row will be skipped (raised when generation has Attempt 1 after Attempt 1 or similar behaviors)

        eval_friendly_d = dict(
            good_solution=good_solution,
            good_ans=get_answer_rims(
                good_solution,
                ans=good_ans,
                method=good_method,
                dataset_type=dataset_type,
            ),
            good_method=good_method,
            bad_solutions=bad_solution,
            bad_ans=[
                get_answer_rims(s, ans=a, method=m, dataset_type=dataset_type)
                for s, a, m in zip(bad_solution, bad_ans, bad_method)
            ],
            bad_method=bad_method,
            mistakes=mistakes,
            hint=hint,
            did_reflect=did_reflect,
        )
        return eval_friendly_d

    rawprompt = open(prompt_f).read().strip()
    prompt_tmp = PromptStr(rawprompt)
    prompt = prompt_tmp.sub("QUESTION", question)  # data['question'])
    assert isinstance(prompt, str)
    messages = [{"role": "user", "content": prompt}]

    # if turn_based:
    #     messages = convert_to_turns(prompt, question)

    # if continue_writing_gpt_messages is not None:
    #     assert isinstance(
    #         continue_writing_gpt_messages, list
    #     ), f"continue_writing_gpt_messages should be a list of messages to openai chat create {continue_writing_gpt_messages=}"
    #     messages.extend(continue_writing_gpt_messages)

    if stop_tok is None:  # decode until it faces correct answer
        stop_tok = [
            "\n`Evaluation`: Correct",
            "`Evaluation`: Correct",
            "Evaluation: Correct",
            "<|end|>",  # phi3
        ]  # could be a list or a single string object. Defaults: None

    # # inspect prompt ready
    # dbgf = prompt_f.replace(".txt", ".jsonl")
    # if not Path(dbgf).exists():
    #     Path(dbgf).parent.mkdir(parents=True, exist_ok=True)
    #     jsl.open(dbgf, "w").write_all(messages + [{"prompt_f": prompt_f}])

    # do query!
    response = query_with_openlimit(  # api_key=key,
        seed=777,
        model=model_name,
        max_tokens=max_tokens,
        stop=stop_tok,
        messages=messages,
        temperature=temperature,
        n=n,
    )
    # completion_usage = response.usage

    # postprocess string out
    if n == 1:
        raw_query_out = response.choices[0].message.content  # str
        # if continue_writing_gpt_messages is not None:
        #     msgs_except_inst = continue_writing_gpt_messages[:-1]
        #     if (
        #         msgs_except_inst
        #     ):  # left outputs to prepend (for the ease of postprocessing (...? maybe?) )
        #         given_as_msgs_str = "\n".join([m["content"] for m in msgs_except_inst])
        #         raw_query_out = given_as_msgs_str + "\n" + raw_query_out
        #         raw_query_out = raw_query_out.strip()
        parsed_dict = parse_raw_modif(raw_query_out)
        try:
            eval_friendly_d = process_rims_out_dict(
                parsed_dict, dataset_type=dataset_type
            )
        except Exception as e:
            print(str(e))
            print(f"{raw_query_out=}")
            print(f"{parsed_dict=}")
            print(
                f"llm_query_utils.query_rims_inference(): failed processing rims output"
            )
            # raise ValueError("llm_query_utils.query_rims_inference(): failed processing rims output")

        return (
            eval_friendly_d,
            parsed_dict,
            raw_query_out,
            messages,
            response,
        )  # completion_usage

    else:  # guess this part is for self-consistency setting of RIMS prompting... should we explore here?
        raw_query_outs = [choice.message.content for choice in response.choices]  # str
        # if continue_writing_gpt_messages is not None:
        #     msgs_except_inst = continue_writing_gpt_messages[:-1]
        #     if (
        #         msgs_except_inst
        #     ):  # left outputs to prepend (for the ease of postprocessing (...? maybe?) )
        #         given_as_msgs_str = "\n".join([m["content"] for m in msgs_except_inst])
        #         raw_query_outs = [
        #             (given_as_msgs_str + "\n" + rqo).strip() for rqo in raw_query_outs
        #         ]
        parsed_dicts = [
            parse_raw_modif(raw_query_out) for raw_query_out in raw_query_outs
        ]
        eval_friendly_ds = [
            process_rims_out_dict(parsed_dict, dataset_type=dataset_type)
            for parsed_dict in parsed_dicts
        ]

        return (
            eval_friendly_ds,
            parsed_dicts,
            raw_query_outs,
            messages,
            response,
        )  # completion_usage


def aggregate_eval_friendly_ds_to_a_dict(
    eval_friendly_ds: List[Dict],
    raw_query_outs: List[str],
    query_msg: List[Dict],
) -> Dict:
    """
    list of eval_friendly_d to a single dict
    used for n>1 setting of query_rims_inference()

    dict(
        good_solution
        good_ans
        good_method
        bad_solutions
        bad_ans
        bad_method
        mistakes
        hint
        did_reflect
        raw_query_out
    )
    """
    result_d = dict()
    result_d["good_solution"] = [d["good_solution"] for d in eval_friendly_ds]
    result_d["good_ans"] = [d["good_ans"] for d in eval_friendly_ds]
    result_d["good_method"] = [d["good_method"] for d in eval_friendly_ds]
    result_d["bad_solutions"] = [d["bad_solutions"] for d in eval_friendly_ds]
    result_d["bad_ans"] = [d["bad_ans"] for d in eval_friendly_ds]
    result_d["bad_method"] = [d["bad_method"] for d in eval_friendly_ds]
    result_d["mistakes"] = [d["mistakes"] for d in eval_friendly_ds]
    result_d["hint"] = [d["hint"] for d in eval_friendly_ds]
    result_d["did_reflect"] = [d["did_reflect"] for d in eval_friendly_ds]
    result_d["raw_query_out"] = raw_query_outs
    result_d["query_msg"] = query_msg
    return result_d


### getting prompts for each method ###
def get_select_prompt2(
    question: str,
    cot_pal_p2c_sln_d: dict = None,
    dataset_type: Literal["gsm", "svamp", "ocw", "math"] = None
    # backbone:str="chatgpt",
) -> List[Dict[str, str]]:
    # open up prompt template yaml file
    prompt_yml = THIS_PARENT / "model_selection_prompts.yaml"
    prompt_d: Dict[str, Any] = yaml.full_load(open(prompt_yml))

    select_prompt_key = f"{dataset_type}_select"
    ds_prom_d: Dict[str, Any] = prompt_d[select_prompt_key]

    # process with the question/solutions of interest to result in gpt_messages
    system: str = ds_prom_d["system"]
    fewshots_user = ds_prom_d["user"]
    fewshots_assistant = ds_prom_d["assistant"]

    # make user's query from (question, cot_pal_p2c_sln_d)
    q = question  # data['question']
    to_replace_keys = "{COT_SOLUTION} {PAL_SOLUTION} {P2C_SOLUTION}"
    user_tmp: str = ds_prom_d["user_tmp"]

    # 1. fill the quesiton
    user_tmp = user_tmp.replace("{QUESTION}", q)
    # 2. fill the solutions
    for to_replace, to_be in zip(to_replace_keys.split(), cot_pal_p2c_sln_d.values()):
        user_tmp = user_tmp.replace(to_replace, to_be)
    user_attempt = user_tmp

    msgs: List[Dict[str, str]] = [
        {"role": "system", "content": system},
    ]
    for fu, fa in zip(fewshots_user, fewshots_assistant):
        usr = {"role": "user", "content": fu}
        astnt = {"role": "assistant", "content": fa}
        msgs.append(usr)
        msgs.append(astnt)

    msgs.append({"role": "user", "content": user_attempt})

    return msgs


def get_select_prompt(
    question: str, cot_pal_p2c_sln_d: dict, backbone: str = "chatgpt"
):
    """
    DEPRECATED and not used (marked on mar18) --> use get_select_prompt2() instead
    This function is used to generate the selection prompt.
    """
    raise ValueError(
        "get_select_prompt() is deprecated. Use get_select_prompt2() instead."
    )
    if len(cot_pal_p2c_sln_d) == 3:
        if backbone.startswith("gpt4"):  # or backbone == "gpt4turbo":
            system_message = math_prompt.GPT4_SELECT_SYSTEM3
            user_message = math_prompt.GPT4_SELECT_USER3
            assistant_message = math_prompt.GPT4_SELECT_ASSISTANT3
        elif backbone.startswith("chatgpt"):
            system_message = math_prompt.TURBO_SELECT_SYSTEM3
            user_message = math_prompt.TURBO_SELECT_USER3
            assistant_message = math_prompt.TURBO_SELECT_ASSISTANT3
    elif len(cot_pal_p2c_sln_d) == 2:
        if backbone.startswith("gpt4"):  # or backbone == "gpt4turbo":
            system_message = math_prompt.GPT4_SELECT_SYSTEM
            user_message = math_prompt.GPT4_SELECT_USER
            assistant_message = math_prompt.GPT4_SELECT_ASSISTANT
        elif backbone.startswith("chatgpt"):
            system_message = math_prompt.TURBO_SELECT_SYSTEM
            user_message = math_prompt.TURBO_SELECT_USER
            assistant_message = math_prompt.TURBO_SELECT_ASSISTANT
    else:
        assert (
            False
        ), f"len(cot_pal_p2c_sln_d) needs to be 2 or 3 (current = {len(cot_pal_p2c_sln_d)})"

    # cot_solution, pal_solution, p2c_solution = cot_pal_p2c_sln_d.values()
    if "cot" in cot_pal_p2c_sln_d.keys():
        cot_solution = cot_pal_p2c_sln_d["cot"]
    if "pal" in cot_pal_p2c_sln_d.keys():
        pal_solution = cot_pal_p2c_sln_d["pal"]
    if "p2c" in cot_pal_p2c_sln_d.keys():
        p2c_solution = cot_pal_p2c_sln_d["p2c"]

    messages = get_user_assistant_messages(
        system_message, user_message, assistant_message
    )

    # clean up pal solution to be free from docstring.
    try:  # looks super unhappy, but look at the code of the original author, they actually did this. [permalink](https://github.com/XuZhao0/Model-Selection-Reasoning/blob/8ee494276e958c3f88332d4be64f8a746395f11c/src/selection_math.py#L100)
        if pal_solution.count('"""') == 2:
            docstr_delim = '"""'
        elif pal_solution.count("'''") == 2:
            docstr_delim = "'''"
        pal_generated_list = pal_solution.split(docstr_delim)
        pal_generated = pal_generated_list[0].strip() + pal_generated_list[2]
    except Exception as e:
        pal_generated = (
            pal_solution[0].strip()
            if isinstance(pal_solution, list)
            else pal_solution.strip()
        )

    if cot_solution[0].startswith(
        "Answer:"
    ):  # put 'Answer:' at the start of CoT answer generation. Original code does this but not sure what they really wanted to do with this... biasing toward CoT?
        cot_generated = (
            cot_solution[0].strip()
            if isinstance(cot_solution, list)
            else cot_solution.strip()
        )
    else:
        cot_generated = (
            "Answer:\n" + cot_solution[0].strip()
            if isinstance(cot_solution, list)
            else "Answer:\n" + cot_solution.strip()
        )

    # if len(cot_pal_p2c_sln_d) == 2:
    user_message = f"""Math problem: {question.strip()}

(A)
{cot_generated.strip()}

(B)
{pal_generated.strip()}

Which of the above two choices can correctly answer the math problem?"""

    # Here we append p2c solution for 3-method case. p2c solution was list, but below, it gets converted to string.
    if len(cot_pal_p2c_sln_d) == 3:
        p2c_choice_str = f"(C)\n{p2c_solution[0].strip() if isinstance(p2c_solution, list) else p2c_solution.strip()}\n\nWhich of the above three choices can correctly answer the math problem?"
        user_message = user_message.replace(
            "Which of the above two choices can correctly answer the math problem?",
            p2c_choice_str,
        )

    messages += [{"role": "user", "content": user_message}]

    return messages


def get_user_assistant_messages(
    system_message: str, user_message: str, assistant_message: str
):
    """
    This function is used to convert the prompt into the message format used by OpenAI Chat API.
    """
    messages = []
    messages.append({"role": "system", "content": system_message})
    split_user_messages = user_message.split("\n" * 4)
    split_assistant_messages = assistant_message.split("\n" * 4)  # delim==4*\n...
    for i in range(
        len(split_user_messages)
    ):  # user messages and assistant messages are paired... actually. This should have been `zip()`.
        question = split_user_messages[i]
        answer = split_assistant_messages[i]
        messages += [
            {"role": "user", "content": f"{question}"},
            {"role": "assistant", "content": f"{answer}"},
        ]
    return messages


def get_cot_prompt(
    question: str,
    backbone: str,
    dataset_type: Literal["gsm", "ocw", "math"] = None,
):
    """
    This function is used to generate the CoT prompt.
    append "Question: " to the `question`
    """
    if dataset_type not in "gsm ocw math":
        raise ValueError(f"get_cot_prompt(): {dataset_type=} is not supported")

    if dataset_type == "gsm":
        if "gpt4" in backbone:  # or backbone == "gpt4turbo":
            system_message = math_prompt.GPT4_COT_SYSTEM
            user_message = math_prompt.GPT4_COT_USER
            assistant_message = math_prompt.GPT4_COT_ASSISTANT
        else:
            system_message = math_prompt.TURBO_COT_SYSTEM
            user_message = math_prompt.TURBO_COT_USER
            assistant_message = math_prompt.TURBO_COT_ASSISTANT
        messages = get_user_assistant_messages(
            system_message, user_message, assistant_message
        )
        messages += [{"role": "user", "content": f"Question: {question}"}]

    elif dataset_type in ["ocw", "math"]:
        # open ocw/MATH targeted CoT prompts
        ymlf = THIS_PARENT / "ocw_MATH_prompts.yaml"
        prompt_d = yaml.full_load(open(ymlf))
        pmpt_d = prompt_d[f"{dataset_type}_cot"]
        system_message = pmpt_d["system"]
        user_msgs = pmpt_d["user"]
        # make it to a chat-history
        assistant_msgs = pmpt_d["assistant"]
        messages = [
            {"role": "system", "content": system_message},
        ]
        assert len(user_msgs) == len(
            assistant_msgs
        ), f"{len(user_msgs)=} should be equal to {len(assistant_msgs)=}"
        for u, a in zip(user_msgs, assistant_msgs):
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        # add question of interest with the template
        user_attempt = pmpt_d["user_tmp"].replace("{QUESTION}", question)
        messages += [{"role": "user", "content": user_attempt}]

    else:
        raise ValueError(f"get_cot_prompt(): {dataset_type=} is not supported")

    return messages


def get_pal_prompt(
    question: str,
    backbone: str,
    dataset_type: Literal["gsm", "ocw", "math", "svamp"] = None,
):
    """
    This function is used to generate the PAL prompt.
    """
    if dataset_type not in "gsm ocw math svamp":
        raise ValueError(f"get_pal_prompt(): {dataset_type=} is not supported")

    if dataset_type in "gsm svamp".split():
        if backbone == "gpt4" or backbone == "gpt4turbo":
            system_message = math_prompt.GPT4_PAL_SYSTEM
            user_message = math_prompt.GPT4_PAL_USER
            assistant_message = math_prompt.GPT4_PAL_ASSISTANT
            messages = get_user_assistant_messages(
                system_message, user_message, assistant_message
            )

            messages += [
                {
                    "role": "user",
                    "content": f"Question: {question}\n\n# solution in Python",
                }
            ]

        else:
            system_message = math_prompt.TURBO_PAL_SYSTEM
            user_message = math_prompt.TURBO_PAL_USER
            assistant_message = math_prompt.TURBO_PAL_ASSISTANT
            messages = get_user_assistant_messages(
                system_message, user_message, assistant_message
            )

            messages += [
                {
                    "role": "user",
                    "content": f"Answer the following question in Python: {question}",
                }
            ]
    elif dataset_type in ["ocw", "math"]:
        # open ocw/MATH targeted CoT prompts
        ymlf = THIS_PARENT / "ocw_MATH_prompts.yaml"
        prompt_d = yaml.full_load(open(ymlf))
        pmpt_d = prompt_d[f"{dataset_type}_pal"]
        system_message = pmpt_d["system"]
        user_msgs = pmpt_d["user"]

        # make it to a chat-history
        assistant_msgs = pmpt_d["assistant"]
        messages = [
            {"role": "system", "content": system_message},
        ]

        assert len(user_msgs) == len(
            assistant_msgs
        ), f"{len(user_msgs)=} should be equal to {len(assistant_msgs)=}"

        for u, a in zip(user_msgs, assistant_msgs):
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})

        # add question of interest with the template
        user_attempt = pmpt_d["user_tmp"].replace("{QUESTION}", question)
        messages += [{"role": "user", "content": user_attempt}]

    else:
        raise ValueError(f"get_pal_prompt(): {dataset_type=} is not supported")

    return messages


def get_plan_prompt(question: str, k_fewshot: int = 0) -> str:
    """
    prep prompt for plan generation
    # put "Question: " in front of the `question`

    """
    PLAN_F = THIS_PARENT / "new_p2c_plan_prompts.yaml"  # "prompts_plan_v2.yaml"
    PLAN_PROMPTS_D = yaml.full_load(open(PLAN_F))
    prompt_d = PLAN_PROMPTS_D

    # q = data['question']
    q = question
    system = prompt_d["system_msg"]
    user_tmp = prompt_d["user_template"]
    user_attempt = user_tmp.replace("{QUESTION}", q)  # f"Question: {q}")

    fewshots_user = prompt_d["fewshots_user"][
        :k_fewshot
    ]  # list of fewshot strings include Question: as a stop sequence.
    fewshots_assistant = prompt_d["fewshots_assistant"][:k_fewshot]

    msgs = [
        {"role": "system", "content": system},
    ]
    for fu, fa in zip(fewshots_user, fewshots_assistant):
        usr = {"role": "user", "content": fu}
        astnt = {"role": "assistant", "content": fa}
        msgs.append(usr)
        msgs.append(astnt)
    msgs.append({"role": "user", "content": user_attempt})

    return msgs


def get_plan2code_prompt(
    question: str,  # data:dict,
    plan: str = "",
    k_fewshot: int = 0,
    custom_idxs: list = None,
):
    """
    prep prompt for code generation
    """
    CODE_F = THIS_PARENT / "new_p2c_code_prompts.yaml"  # "prompts_code_v2.yaml"
    prompt_d = yaml.full_load(open(CODE_F))

    q = question  # data['question']
    system = prompt_d["system_msg"]
    user_tmp = prompt_d["user_template"]
    plan_with_tabs = plan.replace("\n", "\n" + " " * 4)
    user_attempt = user_tmp.replace("{PLAN}", plan_with_tabs).replace("{QUESTION}", q)

    if not custom_idxs:
        fewshots_user = prompt_d["fewshots_user"][
            :k_fewshot
        ]  # list of fewshot strings include Question: as a stop sequence.
        fewshots_assistant = prompt_d["fewshots_assistant"][:k_fewshot]
    else:
        fewshots_user = [prompt_d["fewshots_user"][i] for i in custom_idxs]
        fewshots_assistant = [prompt_d["fewshots_assistant"][i] for i in custom_idxs]

    msgs = [
        {"role": "system", "content": system},
    ]
    for fu, fa in zip(fewshots_user, fewshots_assistant):
        usr = {"role": "user", "content": fu}
        astnt = {"role": "assistant", "content": fa}
        msgs.append(usr)
        msgs.append(astnt)
    msgs.append({"role": "user", "content": user_attempt})

    return msgs


### postprocessing helpers ###
# for p2c response
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
def postprocess_code(rawanswer: str, k_fewshot: int = 0):
    def remove_prints(code: str) -> str:
        lines = code.split("\n")
        lines_ = [
            l if not l.startswith("print(") else l.replace("print(", "# print(")
            for l in lines
        ]
        code_ = "\n".join(lines_)
        return code_

    try:
        # 1 removing starting wrap ```
        if "```python" in rawanswer:
            rawanswer = rawanswer.split("```python")[-1]
        elif rawanswer.startswith("```"):
            rawanswer = rawanswer.split("```")[-1]

        # 2 removing ``` at the end
        code = rawanswer.split("```")[0]  # ending ``` removal

        code = remove_prints(code)
        assert code
    except:
        print("code gen fails (unexecutable or funcname?)")
        print(f"code:\n{rawanswer}")
        code = ""
    return code


# p2c response postprocessing utility
def separate_plan_code(rawstr: str) -> tuple:
    # used for 5_cohlike_prompt
    # p2c results in plan\ncode so split it.
    # new p2c result will not be affected by this. so let it be here still in case of revert
    rawstr = rawstr.strip()
    lines = rawstr.split("\n")
    found_code = False
    for i, l in enumerate(lines):
        if l.startswith("def ") and l.strip().endswith(":"):
            found_code = True
            break
    if found_code:
        plan = "\n".join(lines[:i])
        code = "\n".join(lines[i:])
    else:
        plan, code = None, None
    return plan, code  # plan never used...


# method name normalization for rimsprompt
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


### postprocess backticks for pythoncode output
def parse_python_code_from_string(unparsed_txt: str):
    ptn = r"```python((.|\n)*?)```"
    match = re.search(ptn, unparsed_txt)
    if match is not None:
        return match.group(1)
    else:
        return None


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

    import matplotlib
    import sympy
    import sympy as sp
    from sympy import Symbol
    from sympy import isprime as is_prime
    from sympy import symbols

    matplotlib.use(
        "Agg"
    )  # exec("import matplotlib\nmatplotlib.use('Agg')\n", locals_) # to prevent matplotlib drawing on subthread error

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
            # exec("import matplotlib\nmatplotlib.use('Agg')\n", locals_) # to prevent matplotlib drawing on subthread error
            # raise ValueError("avoid matplotlib (semaphore, NSException)")

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
    if (
        "<|end|>" in code_string
    ):  # for buggy output from tgi-phi3 serving, just let it stay here.
        code_string = code_string.split("<|end|>")[0]

    def get_code_from_backticks_wrap(code_string):
        return postprocess_code(code_string)

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

    # # 0. replace "{somefloatingnumber}$" to {somefloatingnumber} --> problematic
    # ptn = r"(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d*)?\$"
    # cost_usds = re.findall(ptn, solution)
    # for bill in cost_usds:
    #     solution = solution.replace(bill, bill.replace("$", ""))

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
    try:
        ans: str = solution.strip().split("\n")[-1].replace("So the answer is ", "")
        prd: Union[str, None] = _find_the_last_numbers(ans)
        prd = float(prd.replace(",", "").rstrip(".")) if prd else prd
    except Exception as e:
        prd = None
        print(e)
        print("extract_num_turbo")

    return prd


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


def solution2blurb(method: str = "", solution: str = "", ans: Any = ""):
    """
    This function is for `--eval_indiv_method` option in `rims_inference` function
    solution into blurb string
    """
    abbr2full = {
        "cot": "Chain-of-Thought",
        "pal": "Program-aided Language Modeling",
        "p2c": "Plan-and-then-Code",
    }
    blurb_str = f"`Method`: {abbr2full[method]} ({method})\n`Attempt 1`: {solution}\n`Answer 1`: {ans}"
    return blurb_str


def backbone2model(backbone: str) -> str:
    if backbone == "gpt4":
        model_name = "gpt-4"
    elif backbone == "gpt4turbo":  # or backbone == "GPT4-1106":
        # model_name = "gpt-4-1106-preview"
        model_name = "GPT4-1106"
    elif backbone == "chatgpt0613":  # or backbone == "GPT-35":
        model_name = "gpt-3.5-turbo-0613"
    elif backbone == "chatgpt0125":
        # model_name = "gpt-3.5-turbo-0125"
        model_name = "laba-gpt-35-turbo-0125"
    elif backbone == "chatgpt1106":
        # model_name = "gpt-3.5-turbo-1106"
        model_name = "laba-gpt-35-turbo-1106"
    elif backbone == "chatgpt0613long":
        # model_name = "gpt-3.5-turbo-16k-0613"
        model_name = "laba-gpt-35-turbo-16k-0613"
    else:
        model_name = backbone  # openLLM setting
        # raise ValueError(f"backbone: {backbone} is not supported")
    return model_name
