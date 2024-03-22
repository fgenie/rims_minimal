import yaml
from typing import List, Literal
from itertools import permutations
import os
from collections import defaultdict

# from pqdm.processes import pqdm
from tqdm import tqdm
from fire import Fire
from munch import Munch, munchify
from openai import AsyncAzureOpenAI
import asyncio

client = AsyncAzureOpenAI(
        azure_endpoint=os.getenv("OLD_AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("OLD_AZURE_OPENAI_API_KEY"),
        api_version="2023-07-01-preview",
        max_retries=3,
        timeout=120,   
    )

semaphore = asyncio.Semaphore(6)

def flatten(l:List[List[str]])->List[str]:
    return [item for sublist in l for item in sublist]

async def query_openai(
        prompt:List[str], 
        model: str = "GPT4-1106")->List[str]:
        # model: str = "GPT-35")->List[str]:
    async with semaphore:
        response = await client.chat.completions.create(
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
    blurb_placeholder = template.user
    to_replace_map = {
        "QUESTION": question.strip(),
        "WRONG_METHOD": _abb2full(wrong_method),
        "WRONG_SOLUTION": wrong_solution.strip(),
        "WRONG_PRED": wrong_pred.strip(), 
        "CORRECT_METHOD": _abb2full(correct_method),
        "CORRECT_SOLUTION": correct_solution.strip(),
        "CORRECT_PRED": correct_pred.strip(),
    }
    for key, value in to_replace_map.items():
        key_ = f"{{{key}}}"
        blurb_placeholder = blurb_placeholder.replace(key_, str(value))
    prompt.append({"role": "user", "content": blurb_placeholder})

    return prompt, blurb_placeholder
    

async def prep_blurbs(
        src_d: dict, 
        wrong_method: Literal["cot", "pal", "p2c"] = None, 
        correct_method: Literal["cot", "pal", "p2c"] = None,
        progress_bar: tqdm = None,
        ) -> List[str]:
    """
    returns (question * 2) blurbs 
    """    
    
    def _generated2blanks(blurb_w_placeholder:str, bl:str):
        try:
            mistake, hint = [l for l in bl.split("\n") if l.strip()]
            mistake = mistake.replace("`Mistakes`: ", "")
            hint = hint.replace("`Hint for a better Method choice`: ", "")
        except Exception as e:
            mistake, hint = f"ERROR: {bl.strip()}", str(e)
        mistake_placeholder = "<one_liner_explanation_for_whats_gone_wrong_in_the_attempt>"
        hint_placeholder = "<one_liner_hint_to_workaround_with_different_method>"

        filled = blurb_w_placeholder.replace(mistake_placeholder, mistake).replace(hint_placeholder, hint)
        return filled
        

    # extract needs for the prompt:
    # question, wrong_solution, wrong_pred, correct_solution, correct_pred
    all_blurbs = []
    for i, (question) in enumerate(src_d.questions):
        wrong_solution = src_d.wrong[wrong_method][i]
        wrong_pred = src_d.wrong.preds[wrong_method][i]
        correct_solution = src_d.correct[correct_method][i]
        correct_pred = src_d.answers[i]
        prompt, blurb_w_placeholder = get_prompt(
            question = question,
            wrong_method = wrong_method,
            wrong_solution = wrong_solution,
            wrong_pred = wrong_pred,
            correct_method = correct_method,
            correct_solution = correct_solution,
            correct_pred = correct_pred
        )
        two_blurbs_ = await query_openai(prompt)
        two_blurbs = [_generated2blanks(blurb_w_placeholder, bl) for bl in two_blurbs_]

        numbered_blurbs = [f"{i}\n{bl.strip()}" for bl in two_blurbs]
        progress_bar.update(2)
        all_blurbs.extend(numbered_blurbs)

    return all_blurbs


async def main():
    hint_template: dict = yaml.full_load(open("make_blurbs_template.yaml"))
    fewshots: dict = yaml.full_load(open("make_blurbs_resources.yaml"))
    # fewshots: dict = yaml.full_load(open("make_blurbs_resources_pal_p2c_aug.yaml"))

    # make it attribute-accessible
    hint_template = munchify(hint_template)
    fewshots = munchify(fewshots)

    # make the prompt
    # datasets = ["math"]
    datasets = ["ocw", "math"]
    
    # methods = "cot pal p2c".split()
    # wrong2correct_directions = permutations(methods, 2)
    wrong2correct_directions = [
        "p2c-cot", 
        "pal-p2c",  
        "pal-cot", 
        "cot-p2c", 
    ]
    awaitables = []
    keys = []
    pbar = tqdm(
        range(
            len(datasets)* \
            len(wrong2correct_directions)* \
            len(fewshots.math.questions) * 2
            )
    )
    for ds in datasets:
        for wrong_correct in wrong2correct_directions:
            w_method, c_method = wrong_correct.split("-")
            
            # make blurbs
            task = asyncio.create_task(prep_blurbs(fewshots[ds], wrong_method = w_method, correct_method = c_method, progress_bar=pbar))
            awaitables.append(task)
            
            # annotate jobs
            outf = f"blurbs_{ds}_{wrong_correct}.txt"
            # outf = f"blurbs_{ds}_{wrong_correct}_aug.txt"
            keys.append(outf)
            
    # start jobs
    done = await asyncio.gather(*awaitables)

    
    # pair the jobs
    fname2tasks = defaultdict(list)
    for key, tasks in zip(keys, done):
        fname2tasks[key].extend(tasks)

    # write to txt
    for outf, completed in fname2tasks.items():
        with open(outf, "w") as wf:
            sep = f"\n\n\n\n{'='*20}\n\n\n\n"
            for txt in completed:
                wf.write(txt + sep)

    """
    REDUCE the possible candidates we have less time!
    1. best-performed directions and orders (see gsm_best)
    2. take good examples for correct (and wrong too)

    v3 p2c-cot pal-p2c pal-cot
    math 1,2,3    1(weak)  1,2,3   
    ocw  1        1,2,3     3


    # v2 cot-p2c pal-cot pal-p2c # for later use 
    # v1 cot-p2c pal-cot pal-p2c

    """
            
if __name__ == "__main__":
    Fire(main)
            
            
            
        
    