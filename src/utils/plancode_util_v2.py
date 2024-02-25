from typing import Any, Callable, Mapping, Sequence, Union

import yaml

# PLAN_F = "/Users/seonils/dev/llm-reasoners/examples/Model-Selection-Reasoning/src/prompts/prompts_plan_v2.yaml"
# CODE_F = "/Users/seonils/dev/llm-reasoners/examples/Model-Selection-Reasoning/src/prompts/prompts_code_v2.yaml"
PLAN_F = "/Users/seonils/dev/rims_minimal/src/utils/new_p2c_plan_prompts.yaml"
CODE_F = "/Users/seonils/dev/rims_minimal/src/utils/new_p2c_code_prompts.yaml"
from openai import OpenAI, AzureOpenAI


KEY = (
    open(
        "./openai_key.txt"
    )
    .read()
    .strip()
)
# KEY = open("/Users/seonils/my_openai_key.txt").read().strip()
# client = OpenAI(api_key=KEY)
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-07-01-preview",
)


PLAN_PROMPTS_D = yaml.full_load(open(PLAN_F))
CODE_PROMPTS_D = yaml.full_load(open(CODE_F))


def get_plan_prompt(data: dict, k_fewshot: int = 0, hint: str = "") -> str:
    """
    prep prompt for plan generation
    """
    prompt_d = PLAN_PROMPTS_D

    q = data["question"]
    system = prompt_d["system_msg"]
    user_tmp = prompt_d["user_template"]
    if hint:
        user_attempt = user_tmp.replace("{QUESTION}", f"Question: {q} ({hint})")
    else:
        user_attempt = user_tmp.replace("{QUESTION}", f"Question: {q}")

    # print(user_attempt)
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
    data: dict,
    plan: str = "",
    k_fewshot: int = 0,
    custom_idxs: list = None,
    hint: str = "",
):
    # little bit revision from PAL prompt.
    # `solution()` is returned (can execute with solution() call w/o argument
    prompt_d = CODE_PROMPTS_D

    q = data["question"]
    system = prompt_d["system_msg"]
    user_tmp = prompt_d["user_template"]
    if hint:
        q = f"{q} ({hint})"
    plan_with_tabs = plan.replace("\n", "\n"+" "*4)
    user_attempt = user_tmp.replace("{PLAN}", plan_with_tabs).replace(
        "{QUESTION}", f"{q}"
    )
    # print(q)

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


# def postprocess_code_answer(rawanswer:str, docdef:str='', k_fewshot:int=0):
def postprocess_code(rawanswer: str, k_fewshot: int = 0):
    try:
        # 1 removing starting wrap ```
        if "```python" in rawanswer:
            code = rawanswer.split("```python")[-1]
        elif rawanswer.startswith("```"):
            rawanswer = rawanswer.split("```")[-1]

        # 2 removing ``` at the end
        code = rawanswer.split("```")[0]  # ending ``` removal

        # in v1, I tried force decode in prompt which caused so many errors, I will not do it here.
        # if k_fewshot>0: # just use output do not modif
        #     if code.startswith('def solution():'):
        #         pass
        #     else:
        #         code = docdef + '\n' + (code if code.startswith('\t') else f"\t{code}")
        code = remove_prints(code)
        assert code
        # exec(code) # check if it is executable # this is done in tool.py:safe_execute_turbo
    except:
        print("code gen fails (unexecutable or funcname?)")
        print(f"code:\n{rawanswer}")
        code = ""
    return code


def remove_prints(code: str) -> str:
    lines = code.split("\n")
    lines_ = [
        l if not l.startswith("print(") else l.replace("print(", "# print(")
        for l in lines
    ]
    code_ = "\n".join(lines_)
    return code_


def kvprint(record):
    for r in record:
        print(r["role"])
        print(r["content"])


if __name__ == "__main__":

    datas = [
        {"question": "Evaluate $\\left\\lceil -\\frac{7}{4}\\right\\rceil$.", "level": "Level 3", "type": "Algebra", "solution": "$-\\frac{7}{4}$ is between $-1$ and $-2$, so $\\left\\lceil -\\frac{7}{4}\\right\\rceil=\\boxed{-1}$.", "answer": "-1"},
        {"question": "What fraction is the same as \\[\n\\frac{2-4+6-8+10-12+14}{3-6+9-12+15-18+21}?\n\\]", "level": "Level 1", "type": "Algebra", "solution": "We have \\begin{align*}\n&\\frac{2-4+6-8+10-12+14}{3-6+9-12+15-18+21} \\\\\n& \\qquad = \\frac{2(1-2+3-4+5-6+7)}{3(1-2+3-4+5-6+7)} \\\\\n& \\qquad = \\boxed{\\frac{2}{3}}.\n\\end{align*}", "answer": "\\frac{2}{3}"},
        {"question": "What is the domain of the function $$f(x) = \\frac{(2x-3)(2x+5)}{(3x-9)(3x+6)}~?$$ Express your answer as an interval or as a union of intervals.", "level": "Level 5", "type": "Algebra", "solution": "We have $x$ in the domain of $f(x)$ as long as the denominator, $(3x-9)(3x+6)$, is nonzero. This is true for all $x$ except the solutions to the equations $3x-9=0$ and $3x+6=0$. These solutions are $x=3$ and $x=-2$, respectively.\n\nTherefore, the domain of $f(x)$ is all real numbers except $3$ and $-2$. Expressed as a union of intervals, the domain is $\\boxed{(-\\infty,-2)\\cup (-2,3)\\cup (3,\\infty)}$.", "answer": "(-\\infty,-2)\\cup (-2,3)\\cup (3,\\infty)"},
        {"question": "What is the sum of all the odd integers between $500$ and $700$?", "level": "Level 5", "type": "Algebra", "solution": "We want to find the sum of the arithmetic series $501 + 503 + \\dots + 699$.\n\nThe common difference is 2, so the $n^{\\text{th}}$ term in this arithmetic sequence is $501 + 2(n - 1) = 2n + 499$.  If $2n + 499 = 699$, then $n = 100$, so the number of terms in this sequence is 100.\n\nThe sum of an arithmetic series is equal to the average of the first and last term, multiplied by the number of terms, so the sum is $(501 + 699)/2 \\cdot 100 = \\boxed{60000}$.", "answer": "60000"},
        {"question": "If $\\frac{1}{x} + \\frac{1}{y} = \\frac{1}{z}$, find the value of $y$ when $x = 25$ and $z = 24$.", "level": "Level 2", "type": "Algebra", "solution": "We have $\\frac{1}{25} + \\frac{1}{y} = \\frac{1}{24}$, so  \\[\\frac{1}{y} = \\frac{1}{24} - \\frac{1}{25} = \\frac{25}{600} - \\frac{24}{600} = \\frac{1}{600},\\] which means $y=\\boxed{600}$.", "answer": "600"},
    ]
    for data in datas:
        pp2 = get_plan_prompt(data, k_fewshot=8)
        pp2r = client.chat.completions.create(messages=pp2, model="gpt-3.5-turbo", stop="Write a python function to").choices[0].message.content

        cp2 = get_plan2code_prompt(data, plan=pp2r, k_fewshot=8)
        print("===========")
        kvprint(pp2)
        print("===========")
        kvprint(cp2)
        print("===========")
        cp2r = client.chat.completions.create(messages=cp2, model="gpt-3.5-turbo", stop="Question:").choices[0].message.content
        print(cp2r)

        """ # I'm satisfied with the test result below.

        Question: Guesstimate the how many times more the ladies' toilet needed to make the same line length for both gender (it is well-known that the line for the ladies' are much lengthier)?

        Guide:
        1. Start by estimating the length of the line for the ladies' toilet.
        2. Estimate the length of the line for the men's toilet.
        3. Divide the length of the line for the ladies' toilet by the length of the line for the men's toilet.
        4. Round the result to the nearest whole number.
        5. Return the rounded result as the number of times more the ladies' toilet needed to make the same line length for both genders.

        ===========
        def solution(ladies_line_length, mens_line_length):
            times_more = round(ladies_line_length / mens_line_length)
            return times_more
        """
