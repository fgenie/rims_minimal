import yaml
from typing import List, Literal
from itertools import permutations
import os


# from pqdm.processes import pqdm
from tqdm import tqdm
from fire import Fire
from munch import Munch, munchify
from openai import AzureOpenAI

client = AzureOpenAI(
        azure_endpoint=os.getenv("OLD_AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("OLD_AZURE_OPENAI_API_KEY"),
        api_version="2023-07-01-preview",
        max_retries=3,
        timeout=120,   
    )

def query_openai(
        prompt:List[str], 
        model:str="GPT4-1106")->List[str]:

    response = client.chat.completions.create(
        temperature = 0.9,
        model = model,
        messages = prompt,
        seed = 777,
        n = 2,
        max_tokens = 1024,
        stop = "`Workaround Method`:",
    )
    return [response.choices[i].message.content for i in range(len(response.choices))]

def get_prompt(
        question: str,
        wrong_method: Literal["cot", "pal", "p2c"],
        wrong_solution: str,
        wrong_pred: str,
        correct_method: Literal["cot", "pal", "p2c"],
        correct_solution: str,
        correct_pred: str, # answer
        )->List[str]:
    
    def _abb2full(method: str) -> str:
        mapping = {
            "cot": "Chain of Thought (cot)",
            "pal": "Program-aided Language Model (pal)",
            "p2c": "Plan-and-then-Code (p2c)",
        }
        return mapping[method]

    # open template
    template = munchify(
        yaml.full_load(open("make_blurbs_template.yaml"))
        )
    # system 
    system: str = template.system
    prompt = [
        {"role": "system", "content": system},
    ] 
    # content
    content = template.user
    to_replace_map = {
        "QUESTION": question,
        "WRONG_METHOD": _abb2full(wrong_method),
        "WRONG_SOLUTION": wrong_solution,
        "WRONG_PRED": wrong_pred, 
        "CORRECT_METHOD": _abb2full(correct_method),
        "CORRECT_SOLUTION": correct_solution,
        "CORRECT_PRED": correct_pred,
    }
    for key, value in to_replace_map.items():
        key_ = f"{{{key}}}"
        content = content.replace(key_, str(value))
    # print(content)
    prompt.append({"role": "user", "content": content})

    return prompt
    

def prep_blurbs(
        src_d: dict, 
        wrong_method: Literal["cot", "pal", "p2c"] = None, 
        correct_method: Literal["cot", "pal", "p2c"] = None
        ) -> List[str]:
    """
    returns (question * 2) blurbs 
    """    
    
    # extract needs for the prompt:
    # question, wrong_solution, wrong_pred, correct_solution, correct_pred
    all_blurbs = []
    for i, (question) in enumerate(src_d.questions):
        wrong_solution = src_d.wrong[wrong_method][i]
        wrong_pred = src_d.wrong.preds[wrong_method][i]
        correct_solution = src_d.correct[correct_method][i]
        correct_pred = src_d.answers[i]
        prompt = get_prompt(
            question = question,
            wrong_method = wrong_method,
            wrong_solution = wrong_solution,
            wrong_pred = wrong_pred,
            correct_method = correct_method,
            correct_solution = correct_solution,
            correct_pred = correct_pred
        )
        two_blurbs = query_openai(prompt)
        print(two_blurbs[0])
        # print(two_blurbs[1])
        numbered_blurbs = [f"{i}\n{bl.strip()}" for bl in two_blurbs]
        all_blurbs.extend(numbered_blurbs)

    return all_blurbs


def main():
    hint_template: dict = yaml.full_load(open("make_blurbs_template.yaml"))
    fewshots: dict = yaml.full_load(open("make_blurbs_resources.yaml"))

    # make it attribute-accessible
    hint_template = munchify(hint_template)
    fewshots = munchify(fewshots)

    # make the prompt
    datasets = ["ocw", "math"]
    methods = "cot pal p2c".split()
    
    # wrong2correct_directions = permutations(methods, 2)
    wrong2correct_directions = [
        "p2c-cot", # 1
        "pal-p2c", # 3 
        "pal-cot", # 3
        "cot-p2c", # 2
    ]
    for ds in datasets:
        src_d = fewshots[ds]
        for wrong_correct in tqdm(wrong2correct_directions):
            w_method, c_method = wrong_correct.split("-")
            
            # make blurbs
            blurbs = prep_blurbs(src_d, wrong_method = w_method, correct_method = c_method)
            
            # write to txt
            outf = f"blurbs_{ds}_{wrong_correct}.txt"
            with open(outf, "w") as wf:
                sep = f"\n\n\n\n{'='*20}\n\n\n\n"
                for txt in blurbs:
                    wf.write(txt + sep)

    """
    REDUCE the possible candidates we have less time!
    1. best-performed directions and orders (see gsm_best)
    2. take good examples for correct (and wrong too)

    v3 p2c-cot pal-p2c pal-cot
    # v2 cot-p2c pal-cot pal-p2c # for later use 
    # v1 cot-p2c pal-cot pal-p2c

    """
            
if __name__ == "__main__":
    Fire(main)
            
            
            
        
    