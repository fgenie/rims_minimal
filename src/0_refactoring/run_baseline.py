import os
import asyncio
import fire

from run_individual import indiv_query
from task_runner import TaskRunner

import pandas as pd
from typing import  Literal
from pathlib import Path
import json


def dedup(records):
    df = pd.DataFrame(records)
    k = "question" if "question" in df.columns else "problem"
    df = df.drop_duplicates(subset=k)
    records = df.to_dict(orient="records")
    return records


def filter_only_error_rows(err_idxs_f, records, outpath):
    if Path(err_idxs_f).exists() and err_idxs_f:
        assert start_idx == 0, "err_idxs_f is only supported when start_idx is 0"
        idxs = [int(i) for i in open(err_idxs_f).read().strip().split("\n")]
        records = [records[i] for i in idxs]
        outpath = str(outpath).replace(".jsonl", ".jsonl_leftovers")
        while Path(outpath).exists():
            outpath += "_"
        outpath = Path(outpath)

    return records, outpath


def save_res(outpath, res):
    error_rows = []
    with open(outpath, 'w', encoding='utf-8') as f:
        for idx, row in enumerate(res):
            for obj in row.keys():
                if 'error' in row[obj]:
                    error_rows.append(idx)
                    break
            save_keys = ["error", "contents", "query_message", "meta"]
            save_obj = {
                obj: {
                    save_key: row[obj].get(save_key, "")
                    for save_key in save_keys
                }
                for obj in row.keys()
            }
            f.write(json.dumps(save_obj, ensure_ascii=False) + "\n")

    print('Error row count:', len(error_rows))
    print('Error rows:', error_rows)


async def run_task(records, n, temperature, p2c_plan_temperature, backbone, dataset_type, seed, retry=10):
    task_runner_obj = TaskRunner(100, retry=retry)

    for record in records:
        jobs = indiv_query(
            record,
            n=n,
            temperature=temperature,
            p2c_plan_temperature=p2c_plan_temperature,
            seed=seed,
            backbone=backbone,
            dataset_type=dataset_type
        )
        task_runner_obj.add_task(jobs)

    res = await task_runner_obj.run()
    return res


async def main(
    gsm_jslf: str = "",
    dataset_type: Literal[
        "gsm", "ocw", "math"
    ] = "gsm",  # affects get_concordant_answer
    num_methods: int = 3,  # number of methods (3-> cot pal p2c / 2-> cot pal )
    start_idx: int = 0,
    
    # llm options
    n: int = 1,
    backbone: str = "chatgpt0613long",
    seed: int = 777,
    temperature: float = 0.0,
    p2c_plan_temperature: float = 0.0,
    
    # dev option
    retry: int = 10
):
    assert gsm_jslf, f"need to specify {gsm_jslf=}"
    assert dataset_type in "gsm ocw math svamp".split(), f"invalid {dataset_type=}"
    
    import jsonlines
    with jsonlines.open(gsm_jslf) as f:
        records = list(f)[start_idx:]
        records = dedup(records)

    outdir = (
        Path("outputs")
        / f"{Path(gsm_jslf).stem}_dt.{dataset_type}"
        / backbone
    )

    if not outdir.exists():
        outdir.mkdir(parents=True)

    outpath = outdir / f"n{n}_baseline_raw_query_result.jsonl"

    res = await run_task(
        records,
        n,
        temperature,
        p2c_plan_temperature,
        backbone,
        dataset_type,
        seed,
        retry
    )
    save_res(outpath, res)


if __name__ == "__main__":
    fire.Fire(main)