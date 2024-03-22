"""
1. rims prompt
2. rims - hint prompt
3. rims - (wrong + hint) prompt



v3 p2c-cot pal-p2c pal-cot
ocw  0        2     0,1,2   --> 0-2-1 only [one]
math 1,2     aug    1,2     -->  2-aug-1 or 1-aug-2 [two]


# spare blurb to augment above if needed:
      cot-p2c
ocw      2 (allows 1 variant for later if needed)
math     0 (allows 3 variants for later)
"""

import yaml
from munch import munchify
from fire import Fire

from typing import Dict, List
from collections import defaultdict
from pathlib import Path


def ablate(
        blurb: str,
    )->Dict[str,str]:
    # find the end of attempt 1 = mistakes start 
    attempt1_start = blurb.find ("`Attempt 1`: ")
    attempt1_end = blurb.find("`Mistakes`: ")
    
    # find the end of mistakes = hint start
    mistakes_start = attempt1_end
    mistakes_end = blurb.find("`Hint for a better Method choice`: ")    
    
    # find the end of hint = attempt 2 start ( whole reflection ablated)
    hint_start = mistakes_end 
    hint_end = blurb.find("`Workaround Method`: ")
    

    raise NotImplementedError("Ablation not implemented yet")

    ablations = {
        "-hint":, 
        "-hint-mistakes":,
        "-hint-mistakes-attempt1":, 
    }
    return ablations
    
    
     
    



def construct_prompt(
        template_d: Dict[str,str] = None, 
        blurbs_d: Dict[str, List[str]] = None, 
        dirmap: dict = None, 
        blurb_order: List[str] = None
        ) -> str:
    """
    Construct the prompt using the template and blurbs
    """
    # get the template
    system = template_d.system
    # get the blurbs
    blurbs = []
    for direction in blurb_order:
        current_dir_idx = dirmap[direction]
        current_blurb = blurbs_d[current_dir_idx]
        blurbs.append(current_blurb)
    sep = template_d.sep
    
    inst = template_d.instruction 

    all_parts_in_order = [system] + blurbs + [inst]
    
    rims_prompt = sep.join(all_parts_in_order)
    return rims_prompt


def main():
    # predefined mappings of blurbs to use 
    columns = "p2c-cot pal-p2c pal-cot".split()
    prompt_mappings: Dict[List] = {
        "ocw": [
                dict(zip(columns, [0,2,1]))
                ],
        "math": [
                dict(zip(columns, [2,0,1])), 
                dict(zip(columns, [1,0,2])),
                ]
    }

    # load blurbs
    blurbs_d = yaml.full_load(open("math_ocw_selected_blurbs.yaml"))
    blurbs_d = munchify(blurbs_d)

    # load template
    template_d = yaml.full_load(open("make_blurbs2prompt.yaml"))
    template_d = munchify(template_d)
    
    # make the prompt
    for ds, dirmappings_list in prompt_mappings.items():
        for dirmap in dirmappings_list: 
            prompt: str = construct_prompt(template_d, blurbs_d[ds], dirmap, blurb_order = columns)    
            outf = f"rims_{ds}_{'.'.join(columns)}.txt"
            if Path(outf).exists():
                outf = outf.replace(".txt", ".txt1")
            with open(outf, "w") as writer:
                writer.write(prompt)
                print(outf)


if __name__ == "__main__":
    Fire(main)