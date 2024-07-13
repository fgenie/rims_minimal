from pathlib import Path
from typing import List, Literal

import jsonlines as jsl
from fire import Fire
from processings.text_exec_functions import (
    extract_ans_from_cot_MATHnOCW,
    extract_num_turbo,
    get_concordant_answer,
    get_concordant_answer_n,
    safe_execute_turbo,
)
from processings.text_parse_functions import postprocess_code
from tqdm import tqdm


def process_indiv(
    infile: str = "raw_indiv.jsonl",
    outfile: str = "processed_indiv.jsonl",
):
    """
    processed_indiv.jsonl
    row:
    {
    "question":str,
    "answer":str,

    "cot_pred": List,
    "pal_pred": List,
    "p2c_pred": List,
    "majvote_answers": List,

    "cot_solution": List[str],
    "pal_solution": List[str],
    "p2c_solution": List[str],
    }

    """
    parent_dir = Path(infile).parent
    print(f"parent dir is automatically set to {parent_dir=}")
    print(f"outfile is set to {Path(outfile).name} and saved under {parent_dir=}")
    infile = Path(infile).name
    outfile = Path(outfile).name
    assert infile != outfile

    records = list(jsl.open(parent_dir / infile))

    dataset_type = records[0]["CoTQueryObject"]["meta"]["dataset_type"]
    processed_rows = []
    for row in tqdm(records):
        # question = row["question"]
        question = row["CoTQueryObject"]["query_message"][-1]["content"].replace(
            "Question: ", ""
        )
        # answer = row["answer"]

        # regardless of n,
        raw_cots = row["CoTQueryObject"]["contents"]
        raw_pals = row["PALQueryObject"]["contents"]
        raw_p2cs = row["P2CQueryObject"]["contents"]

        # solutions
        cot_solutions = raw_cots
        pal_solutions = [postprocess_code(r) for r in raw_pals]
        p2c_solutions = [postprocess_code(r) for r in raw_p2cs]

        # executions
        cot_exec = (
            extract_num_turbo
            if dataset_type == "gsm"
            else extract_ans_from_cot_MATHnOCW
        )
        code_exec = safe_execute_turbo
        cot_preds = [cot_exec(s) for s in cot_solutions]
        pal_preds = [code_exec(s) for s in pal_solutions]
        p2c_preds = [code_exec(s) for s in p2c_solutions]

        majvote_answers = [
            get_concordant_answer([c, p, p2], dataset_type=dataset_type)
            for c, p, p2 in zip(cot_preds, pal_preds, p2c_preds)
        ]  # List[Union[str,float,None]]

        # to run selection
        need_selection = [maj is None for maj in majvote_answers]

        # for later ease of scoring
        gt_answer = row["CoTQueryObject"]["meta"]["gt_answer"]

        processed_row = dict(
            question=question,
            # answer = answer,
            cot_solutions=cot_solutions,
            pal_solutions=pal_solutions,
            p2c_solutions=p2c_solutions,
            cot_preds=cot_preds,
            pal_preds=pal_preds,
            p2c_preds=p2c_preds,
            majvote_answers=majvote_answers,
            need_selection=need_selection,
            dataset_type=dataset_type,
            gt_answer=gt_answer,
        )
        processed_rows.append(processed_row)

    outjslf = Path(parent_dir) / outfile
    if not outjslf.parent.is_dir():
        outjslf.parent.mkdir(parents=True, exist_ok=True)

    with jsl.open(outjslf, "w") as writer:
        writer.write_all(processed_rows)
        print(f"wrote {len(processed_rows)} rows to")
        print("\t", outjslf)


def process_simple_greedy(
    infile: str = "",
    outfile: str = "processed_sg.jsonl",
    n: int = 1,
):
    assert infile, f"need to specify {infile=}"

    # path
    parent_dir = Path(infile).parent
    print(f"parent dir is automatically set to {parent_dir=}")
    print(f"outfile is set to {Path(outfile).name} and saved under {parent_dir=}")
    infile = Path(infile).name
    outfile = Path(outfile).name
    assert infile != outfile

    if n > 1 or not infile.startswith("n1_"):
        raise NotImplementedError("n>1 cannot run here")

    raw_selections = list(jsl.open(parent_dir / infile))
    selidxs = [
        int(k)
        for k in open(parent_dir / "selection_performed_idxs.txt")
        .read()
        .strip()
        .split("\n")
    ]

    assert len(raw_selections) == len(selidxs)
    originals = list(jsl.open(parent_dir / infile.replace(".jsonl", "_input.jsonl")))

    # process
    import re

    def _find_first_abc(text):
        match = re.search(r"\((A|B|C)\)", text)
        return match.group() if match else None

    def _process_sg_raw(sel: dict = None, og: dict = None) -> dict:
        # read raw sel row and fill the og row with it.
        abc = _find_first_abc(sel["ModelSelectionQuery"]["contents"][0])
        if abc is None:
            og["sg_answer"] = [None]  # failed to generate the selection
            og["sg_selected"] = "failed"
        elif abc == "(A)":
            og["sg_answer"] = og["cot_preds"]  # list
            og["sg_selected"] = "cot"  # list
        elif abc == "(B)":
            og["sg_answer"] = og["pal_preds"]
            og["sg_selected"] = "pal"
        elif abc == "(C)":
            og["sg_answer"] = og["p2c_preds"]
            og["sg_selected"] = "p2c"
        else:
            og["sg_answer"] = [None]  # failed to generate the selection
            og["sg_selected"] = "failed"

        return og

    for idx, selrow in zip(selidxs, raw_selections):
        originals[idx] = _process_sg_raw(sel=selrow, og=originals[idx])

    # save
    outjslf = Path(parent_dir) / outfile
    if not outjslf.parent.is_dir():
        outjslf.parent.mkdir(parents=True, exist_ok=True)

    with jsl.open(outjslf, "w") as writer:
        writer.write_all(originals)
        print("\t", outjslf)


if __name__ == "__main__":
    Fire()
