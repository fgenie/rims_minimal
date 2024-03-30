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
from collections import defaultdict, OrderedDict
from pathlib import Path
from itertools import permutations


def ablate(
        blurb: str,
    )->Dict[str,str]:
    """
    blurb text is as follows, we will ablate 
    (1) -hint,
    (2) -hint -mistakes,
    (3) -hint -mistakes -attempt1

    `Question`: 
    {QUESTION}
    `Method`: {WRONG_METHOD}
    `Attempt 1`: 
    {WRONG_SOLUTION}
    `Answer 1`: {WRONG_PRED}
    `Evaluation`: Wrong
    `Mistakes`: <one_liner_explanation_for_whats_gone_wrong_in_the_attempt>
    `Hint for a better Method choice`: <one_liner_hint_to_workaround_with_different_method>
    `Workaround Method`: {CORRECT_METHOD}
    `Attempt 2`: 
    {CORRECT_SOLUTION}
    `Answer 2`: {CORRECT_PRED}
    `Evaluation`: Correct
    
    """
    # (1) -hint
    hint_start = blurb.find("`Hint for a better Method choice`: ")
    hint_end = blurb.find("`Workaround Method`: ")
    blurb_h = blurb[:hint_start] + blurb[hint_end:]
    
    # (2) -hint -mistakes
    mistake_start = blurb_h.find("`Mistakes`: ")
    mistake_end = blurb_h.find("`Workaround Method`: ")
    blurb_h_m = blurb_h[:mistake_start] + blurb_h[mistake_end:]
    
    # (3) -hint -mistakes -attempt1
    attempt1_start = blurb_h_m.find("`Method`: ")
    attempt1_end = blurb_h_m.find("`Workaround Method`: ")
    blurb_h_m_a1 = blurb_h_m[:attempt1_start] + blurb_h_m[attempt1_end:]
    blurb_h_m_a1 = blurb_h_m_a1.replace(
        "`Workaround Method`: ", "`Method`: ").replace(
        "`Attempt 2`: ", "`Attempt 1`: ").replace(
        "`Answer 2`: ", "`Answer 1`: ")
    



    ablations = {
        "_": blurb,
        "-hint": blurb_h, 
        "-hint-mistakes": blurb_h_m,
        "-hint-mistakes-attempt1": blurb_h_m_a1, 
    }
    return ablations
    
    
     
    



def construct_prompt(
        template_d: Dict[str,str] = None, 
        blurbs_d: OrderedDict[str, List[str]] = None, 
        dirmap: dict = None, 
        blurb_order: List[str] = None
        ) -> str:
    """
    Construct the prompt using the template and blurbs
    """
    # get the template
    system = template_d.system
    # get the blurbs
    prompts_d = defaultdict(list)
    assert blurb_order == list(dirmap.keys())
    for direction in blurb_order:
        current_dir_idx = dirmap[direction]
        current_blurb = blurbs_d[direction][current_dir_idx].strip()
        abl_blurbs: Dict[str,str] = ablate(current_blurb)
        for k,v in abl_blurbs.items():
            prompts_d[k].append(v)
    
    sep = template_d.sep
    inst = template_d.instruction 

    # all parts ready to be joined with sep
    for k, v in prompts_d.items():
        all_parts_to_join = [system] + prompts_d[k] + [inst]
        rims_prompt = sep.join(all_parts_to_join)
        prompts_d[k] = rims_prompt

    return prompts_d


def find_possible_dir2idxs_mappings(blurbs_d, already_done) -> Dict[str, List]:
    """
    Find the possible mappings of directions to indices
    """
    possible_prompt_mappings = dict()
    for ds, directions in blurbs_d.items():
        all_orders = list( permutations(directions, 3) )
        all_idxs = list( permutations(range(3), 3) )
        all_dir2idx_mapping = [OrderedDict(zip(order, idxs)) for order, idxs in zip(all_orders, all_idxs)]
        filtered = [od for od in all_dir2idx_mapping if od not in already_done[ds] ]
        possible_prompt_mappings[ds] = filtered
    return possible_prompt_mappings



def main(
):
    # predefined mappings of blurbs to use 

    # load blurbs
    blurbs_d = yaml.full_load(open("math_ocw_selected_blurbs_mar29.yaml"))
    blurbs_d = munchify(blurbs_d)


    already_cols = "p2c-cot pal-p2c pal-cot".split()
    already_done: Dict[str, List] = {
        "ocw": [
                OrderedDict(zip(already_cols, [0,2,1]))
                ],
        "math": [
                OrderedDict(zip(already_cols, [2,0,1])), 
                OrderedDict(zip(already_cols, [1,0,2])),
                ]
    }

    prompt_mappings = find_possible_dir2idxs_mappings(blurbs_d, already_done)



    

    # load template
    template_d = yaml.full_load(open("make_blurbs2prompt.yaml"))
    template_d = munchify(template_d)
    
    # make the prompt
    for ds, dirmappings_list in prompt_mappings.items():
        for dirmap in dirmappings_list: 
            order = list(dirmap.keys())
            try:
                prompt_d: Dict[str, str] = construct_prompt(template_d, blurbs_d[ds], dirmap, blurb_order=order)    
            except Exception as e:
                print(e)
                print(f"Skipping \n{ds} \n{dirmap}")
                continue
            for abl_setting, prompt_str in prompt_d.items():
                outf = f"prompts/rims_mar29_{ds}_{'.'.join(order)}_{abl_setting}.txt"
                if not Path(outf).parent.exists():
                    Path(outf).parent.mkdir(parents=True)
                if Path(outf).exists():
                    outf = outf.replace(".txt", ".txt1")
                with open(outf, "w") as writer:
                    writer.write(prompt_str)
                    print(outf)


if __name__ == "__main__":
    Fire(main)