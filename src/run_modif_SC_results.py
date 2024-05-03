# this will process SC=15 result into SC < 15 results and save them into a jsonlines file.
# or merge SC<15 results into SC=15 results and save them into a jsonlines file.
# saving query costs and time

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Union

import jsonlines as jsl
import pandas as pd
import yaml
from fire import Fire

from utils.llm_query_utils import get_concordant_answer_n

"""
{
  "index": 1111,
  "question": "There are currently 3 red balls, 11 blue balls, and 25 green balls in the store. Red balls cost $9, Blue balls cost $5 and green balls cost $3. How much will the store have received after all the balls are sold?",
  "answer": 157,
  "ansmap": {
    "cot": [
      157,
      157,
      157,
      157,
      157
    ],
    "pal": [
      157,
      157,
      157,
      157,
      157
    ],
    "p2c": [
      157,
      157,
      157,
      157,
      157
    ]
  },
  "solmap": {
    "cot": [
    ],
    "pal": [
    ],
    "p2c": [
    ]
  },
  "plan": [
  ],
  "error": false,
  "error_msg": "",
  "running_at": "baseline_complete_row",
  "majority_ans": 157,
  "idx2chosen_method": {},
  "majvote_ans": [
  ],
  "candid_answers": [
  ],
  "inference_mode": [
  ],
  "dataset_type": "gsm",
  "prompt_file": "prompt_construction_src/newer_prompts_3/model_selection_prompts.yaml",
  "temperatures": {
    "cot_temperature": 0.5,
    "pal_temperature": 0.8,
    "n": 5
  }
}

plan : just one per file, list
ansmap: dict[list]
solmap: dict[list]
majvote_ans: list
candid_answers: list
inference_mode: list
"""


def merge_pieces(
    records1: List[Dict],
    records2: List[Dict],
    # dataset_type: Literal["gsm", "ocw", "math"],
) -> List[Dict]:
    # check for length and match each row (almost no use for stable result w/o errors )
    keys_of_interest = ["majvote_ans", "candid_answers", "inference_mode", "plan"]
    keys_d_of_interest = ["solmap", "ansmap"]

    df1, df2 = map(pd.DataFrame, [records1, records2])
    question_key = "question" if "question" in df1.columns else "problem"
    df1.drop_duplicates(subset=[question_key], inplace=True)
    df2.drop_duplicates(subset=[question_key], inplace=True)
    df1.rename(columns={"runnning_at": "running_at"}, inplace=True)
    df2.rename(columns={"runnning_at": "running_at"}, inplace=True)
    df1 = df1.dropna(subset=keys_of_interest + keys_d_of_interest)
    df2 = df2.dropna(subset=keys_of_interest + keys_d_of_interest)

    records1 = df1.to_dict(orient="records")
    records2 = df2.to_dict(orient="records")

    if len(df1) != len(df2):  # disparity exists
        if len(df1) < len(df2):
            shorter = df1
            longer = df2
        else:  # >, equality is not here
            shorter = df2
            longer = df1

        # filter longer with shorter
        delta = len(longer) - len(shorter)
        longer = longer[longer[question_key].isin(shorter[question_key])]
        shorter = shorter[shorter[question_key].isin(longer[question_key])]
        print(delta, "rows dropped")

        records1 = longer.to_dict(orient="records")
        records2 = shorter.to_dict(orient="records")

    # sort those
    records1 = sorted(records1, key=lambda row: row[question_key])
    records2 = sorted(records2, key=lambda row: row[question_key])

    def _merge_dict(dict1, dict2):
        assert dict1.keys() == dict2.keys(), "key disparity"
        for k, v in dict1.items():
            dict2[k].extend(v)
        return dict2

    # start merging
    merged_records = []
    keys_of_interest = ["majvote_ans", "candid_answers", "inference_mode", "plan"]
    for row1, row2 in zip(records1, records2):
        # check disparity
        assert row1[question_key] == row2[question_key], "question disparity"
        assert row1["dataset_type"] == row2["dataset_type"], "dataset_type disparity"
        if row1["running_at"] == "rims_complete_row":
            assert row2["running_at"] == row1["running_at"], "running_at disparity"
            assert row2["prompt_file"] == row1["prompt_file"], "prompt file disparity"

        # start merging
        merged_row = deepcopy(row1)

        # merge ansmap, solmap
        if isinstance(row1["ansmap"], dict) and isinstance(row2["ansmap"], dict):
            merged_row["ansmap"] = _merge_dict(row1["ansmap"], row2["ansmap"])
            merged_row["solmap"] = _merge_dict(row1["solmap"], row2["solmap"])
        else:
            continue
        merged_row["temperatures"]["n"] = 15
        if "eval_friendly_d_" in row1.keys():
            if not "eval_friendly_d_" in row2.keys():
                continue
            # extend
            merged_row["eval_friendly_d_"] = _merge_dict(
                row1["eval_friendly_d_"], row2["eval_friendly_d_"]
            )
            # update idx2chosen_method
            updated_idx2chosen_method = row2["idx2chosen_method"]
            offset = row1["temperature"]["n"]
            updated_idx2chosen_method = {
                str(int(k) + offset): v for k, v in updated_idx2chosen_method.items()
            }
            merged_row["idx2chosen_method"].update(updated_idx2chosen_method)
            # update temperatures
            merged_row["temperatures"]["n_adj"] = (
                row1["temperatures"]["n_adj"] + row2["temperatures"]["n_adj"]
            )

        # merge other fields
        for k in keys_of_interest:
            if k in row1.keys() and k in row2.keys():
                merged_row[k] = row1[k] + row2[k]

        # determine majority answer
        assert (
            len(merged_row["candid_answers"]) == 15
        ), f"check whether {keys_of_interest} are properly extended\n{len(merged_row['candid_answers'])=}"
        if "eval_friendly_d_" in merged_row.keys():
            assert (
                len(merged_row["eval_friendly_d_"]["good_method"]) == 15
            ), f"check whether {keys_of_interest} are properly extended\n{len(merged_row['eval_friendly_d_'])=}"

        dataset_type = merged_row["dataset_type"]
        assert row1["dataset_type"] == row2["dataset_type"], "dataset_type disparity"
        majority_ans = get_concordant_answer_n(
            merged_row["candid_answers"], dataset_type=dataset_type
        )
        merged_row["majority_ans"] = majority_ans

        # append
        merged_records.append(merged_row)

    return merged_records


def split_SC15_to_5_10(records: List[Dict]) -> Dict[str, List[Dict]]:
    def process_row(row):
        sc5row, sc10row = deepcopy(row), deepcopy(row)

        # list field split
        keys_of_interest = ["majvote_ans", "candid_answers", "inference_mode"]
        for k in keys_of_interest:
            sc5row[k] = sc5row[k][:5]
            sc10row[k] = sc10row[k][5:]

        # dict field split (ansmap, solmap)
        sc5row["ansmap"] = {k: v[:5] for k, v in sc5row["ansmap"].items()}
        sc5row["solmap"] = {k: v[:5] for k, v in sc5row["solmap"].items()}
        sc10row["ansmap"] = {k: v[5:] for k, v in sc10row["ansmap"].items()}
        sc10row["solmap"] = {k: v[5:] for k, v in sc10row["solmap"].items()}
        sc5row["temperatures"]["n"] = 5
        sc10row["temperatures"]["n"] = 10
        if "eval_friendly_d_" in sc5row and "eval_friendly_d_" in sc10row:
            sc5row["eval_friendly_d_"] = {
                k: v[:5] for k, v in sc5row["eval_friendly_d_"].items()
            }
            sc10row["eval_friendly_d_"] = {
                k: v[5:] for k, v in sc10row["eval_friendly_d_"].items()
            }
            # update idx2chosen_method
            sc5row["idx2chosen_method"] = {
                k: v for k, v in sc5row["idx2chosen_method"].items() if int(k) < 5
            }
            sc10row["idx2chosen_method"] = {
                str(int(k) - 5): v
                for k, v in sc10row["idx2chosen_method"].items()
                if int(k) >= 5
            }
            # update temperatures
            sc5row["temperatures"]["n_adj"] = sc5row["majvote_ans"].count(None)
            sc10row["temperatures"]["n_adj"] = sc10row["majvote_ans"].count(None)

        # determine majority_ans
        sc5row["majority_ans"] = get_concordant_answer_n(
            sc5row["candid_answers"], dataset_type=sc5row["dataset_type"]
        )
        sc10row["majority_ans"] = get_concordant_answer_n(
            sc10row["candid_answers"], dataset_type=sc10row["dataset_type"]
        )

        return sc5row, sc10row

    records_SC5, records_SC10 = [], []
    # filter out none
    df = pd.DataFrame(records)
    L = len(df)
    df.dropna(subset=["majvote_ans", "candid_answers", "inference_mode"], inplace=True)
    df.dropna(subset=["ansmap", "solmap"], inplace=True)
    delta = L - len(df)
    records = df.to_dict(orient="records")
    print(delta, "rows dropped")
    for row in records:
        sc5row, sc10row = process_row(row)
        records_SC5.append(sc5row)
        records_SC10.append(sc10row)

    return {"SC5": records_SC5, "SC10": records_SC10}


def main(ymlf: str = "run_modif_SC_results.yaml"):
    with open(ymlf) as yml:
        yml_dict = yaml.full_load(yml)
    # merge
    yml_to_merge_d: Dict[str, List[str]] = yml_dict["to_merge"]

    # determine which to be done: split? or merge?
    for wfname, pair in yml_to_merge_d.items():
        records1 = list(jsl.open(pair[0]))
        records2 = list(jsl.open(pair[1]))
        merged_records = merge_pieces(records1, records2)
        with jsl.open(wfname, "w") as writer:
            writer.write_all(merged_records)
            print(wfname)
            print(len(merged_records))

    # split
    yml_to_split_d: List[str] = yml_dict["to_split"]

    for fname_tmp, tospl_f in yml_to_split_d.items():
        assert (
            "_scXX" in fname_tmp
        ), f"fname_tmp should contain _scXX, check {ymlf}, \n\n{yml_to_split_d.keys()=}"
        records = list(jsl.open(tospl_f))
        split_records_d = split_SC15_to_5_10(records)
        sc5_records = split_records_d["SC5"]
        sc10_records = split_records_d["SC10"]
        sc5name, sc10name = fname_tmp.replace("XX", "5"), fname_tmp.replace("XX", "10")
        with jsl.open(sc5name, "w") as writer5, jsl.open(sc10name, "w") as writer10:
            writer5.write_all(sc5_records)
            writer10.write_all(sc10_records)
            print(sc5name, len(sc5_records))
            print(sc10name, len(sc10_records))


if __name__ == "__main__":
    Fire(main)
