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
    exp_dir: str = "",
    dataset_type: Literal["gsm", "ocw", "math"] = "",
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
    "p2c_plan": List[str],
    }

    """
    rawjslf = Path(exp_dir) / infile
    records = list(jsl.open(rawjslf))

    processed_rows = []
    for row in tqdm(records):
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

        processed_row = dict(
            cot_solutions=cot_solutions,
            pal_solutions=pal_solutions,
            p2c_solutions=p2c_solutions,
            cot_preds=cot_preds,
            pal_preds=pal_preds,
            p2c_preds=p2c_preds,
            majvote_answers=majvote_answers,
            need_selection=need_selection,
        )
        processed_rows.append(processed_row)
    with jsl.open(exp_dir / outfile, "w") as writer:
        writer.write_all(processed_rows)
        print(f"wrote {len(processed_rows)} rows to")
        print("\t", str(exp_dir / outfile))


def process_simple_greedy(
    exp_dir: str = "",
    infile: str = "raw_simple_greedy.jsonl",
    outfile: str = "processed_simple_greedy.jsonl",
):
    rawjslf = Path(exp_dir) / "raw_simple_greedy.jsonl"
    records = list(jsl.open(rawjslf))
    for row in tqdm(records):
        if set(row["majvote_ans"]) == {None}:  # all None
            continue
