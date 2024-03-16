import tiktoken
import pandas as pd 
import jsonlines as jsl 
from pathlib import Path


encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens



# v1 prompt token = 2031
# v2 prompt token = 2610
# v3 prompt token = 3044

files = Path().glob("*rm_ans")
with open("token_counts_rims3_prompts.txt", "w") as wf:
    for f in files:
        print(str(f.name), file=wf)
        print(num_tokens_from_string(f.open().read()), file=wf)