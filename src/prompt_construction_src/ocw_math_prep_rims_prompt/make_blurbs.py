import yaml
from typing import List
from itertools import permutations


from pqdm.processes import pqdm
from fire import Fire
from munch import Munch, munchify
from openai import AzureOpenAI

client = AzureOpenAI(
        azure_endpoint=os.getenv("OLD_AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("OLD_AZURE_OPENAI_API_KEY"),
        api_version="2023-07-01-preview",
    )

def query_openai(
        prompt:List[str], 
        model:str="GPT4-1106")->str:
    response = client.chat.completions.create(
        temperature = 0.9,
        model = model,
        messages = msgs,
        seed = 777,
        max_tokens = 200,
    )
    return response.choices[0].message.content


def main():
    hint_template: dict = yaml.full_load(open("make_blurbs_template.yaml"))
    fewshots: dict = yaml.full_load(open("make_blurbs_resources.yaml"))

    # make it attribute-accessible
    hint_template = munchify(hint_template)
    fewshots = munchify(fewshots)

    # make the prompt
    datasets = ["ocw", "math"]
    methods = "cot pal p2c".split()
    wrong2correct_directions = permutations(methods, 2)
    for ds in datasets:
        src_d = fewshots[ds]
        for wrong_correct in wrong2correct_directions:
            w_method, c_method = wrong_correct
        
    

    """
    REDUCE the possible candidates we have less time!
    1. best-performed directions and orders (see gsm_best)
    2. take good examples for correct (and wrong too)
    """