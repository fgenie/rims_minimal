from datetime import datetime
from functools import partial
from typing import Any, Literal

import jsonlines as jsl
import pandas as pd
from fire import Fire
from pqdm.processes import pqdm
from tqdm import tqdm

from utils.llm_query_utils import *
from utils.llm_query_utils import _query # explicit import needed for hidden.

# CONTINUE_WRITING_INVOKE_PROMPT = "Continue reaching to the correct answer, carefully following the format presented above."  # same as in 5_extend_*.py

"""
unify the field of the inference records.

row:
    (from gsm)
    - idx
    - question
    - ans
    (from what's inferred)
    - prompt_path
    - ansmap:
        {cot: pal: p2c:}
    - solmap:
        {cot: pal: p2c:}
    - majority_ans: None or majority answer (final answer to be evaluated)
    - selection_or_rims:
        각자에 합당한 아웃풋
        rims: eval_friendly_d
        model_selection (baseline):
            {
                good_method,
                good_ans,
                selection_choice,
                bad_method,
                bad_ans,
            }
        majority_vote: True (if inference ends with majority vote)



"""

# def split_records_into_chunks(records: list, chunksize:int=30)->list:
#     """
#     split records into chunks of size `chunksize`
#     """
#     return [records[i:i+chunksize] for i in range(0, len(records), chunksize)]


def indiv_inference(
    row: dict = None,
    num_methods: int = 3,
    temperature: float = 0.0,
    n: int = 1,
    seed: int = 777,
    backbone: str = "chatgpt0613long", 
    dataset_type: Literal["gsm", "ocw", "math"] = "",
):
    """
    inference each method and return indiv results
    if there are already existing results, use them.


    return:
        solmap : {cot: pal: p2c:}
        ansmap : {cot: pal: p2c:} (solution executed)
    """

    if n > 1:
        raise NotImplementedError(
            "n>1 will serve as a self-consistency parameter, not implemented yet"
        )
    if dataset_type == "ocw":
        question = row["problem"]
    else:
        question = row["question"]

    # check for already-done indiv methods
    if "ansmap" in row.keys() and "solmap" in row.keys():
        if row["ansmap"] and row["solmap"]:
            ansmap = row["ansmap"]
            solmap = row["solmap"]
            missing_methods = []
            for method in ["cot", "pal", "p2c"]:
                if method not in ansmap.keys():
                    if not ansmap[method]:
                        missing_methods.append(method)
    else:
        missing_methods = "cot pal p2c".split()
        ansmap = dict()
        solmap = dict()
    
    # check cot already exists or do query
    if "cot" in missing_methods:
        cot_lst, _msgs, _ = query_cot(
            question, dataset_type=dataset_type,
            temperature=temperature, n=n, backbone=backbone, seed=seed
        )
        # dbgf=f"cot_debug_{dataset_type}.jsonl"
        # if not Path(dbgf).exists(): 
        #     jsl.open(dbgf, "w").write_all(_msgs)
        cot_sol = cot_lst.pop()  # solution: str
        if dataset_type in "gsm":
            cot_ans = extract_num_turbo(cot_sol)
        elif dataset_type in "ocw math":
            cot_ans = extract_ans_from_cot_MATHnOCW(cot_sol)
        else:
            raise ValueError(f"unsupported dataset_type: {dataset_type}")
        solmap["cot"] = cot_sol
        ansmap["cot"] = cot_ans
    else:
        cot_ans = ansmap["cot"]
        cot_sol = solmap["cot"] 
    
    # check pal already exists or do query
    if "pal" in missing_methods:
        pal_lst, __msgs, _ = query_pal(
            question, temperature=temperature, n=n, backbone=backbone, seed=seed, dataset_type=dataset_type
        )
        # dbgf=f"pal_debug_{dataset_type}.jsonl"
        # if not Path(dbgf).exists(): 
        #     jsl.open(dbgf, "w").write_all(__msgs)
        pal_sol = pal_lst.pop()
        pal_ans = safe_execute_turbo(pal_sol)
        solmap["pal"] = pal_sol
        ansmap["pal"] = pal_ans
    else:
        pal_ans = ansmap["pal"]
        pal_sol = solmap["pal"]

    plan = None # for num_methods ==2, plan: null
    if num_methods == 3:
        # check p2c already exists or do query
        if "p2c" in missing_methods:
            
            # try:
            code_lst, plan_lst, ___msgs = query_plancode(
                question,
                plan_temperature=temperature,
                code_temperature=temperature,
                backbone=backbone,
                n=n,
                seed=seed,
            )
            # dbgf=f"p2c_debug_{dataset_type}.jsonl"
            # if not Path(dbgf).exists():
            #     msgs = ___msgs["planquery"]  + ___msgs["codequery"] 
            #     jsl.open(f"p2c_debug_{dataset_type}.jsonl", "w").write_all(msgs)

            plan = plan_lst.pop()
            p2c_solution = code_lst # plan is now generated inbetween the docstring
            if code_lst:
                code = code_lst[0]
            if code is not None:
                p2c_ans = safe_execute_turbo(code)
            else:
                p2c_ans = None

            ansmap["p2c"] = p2c_ans
            solmap["p2c"] = p2c_solution
        else:
            p2c_ans = ansmap["p2c"]
            p2c_solution = solmap["p2c"]
            plan = row["plan"]
    # heuristic for MATH/OCW problematic answer by PAL/P2C (never-endingly long execution result)            
    # OCW max answer length = 210; MATH max answer length = 81
    if len(str(pal_ans)) > 400: # over 400 --> truncate and string return
        print(f"truncating PAL result to 400 chars ({len(pal_ans)=})")
        pal_ans = pal_ans[:400]
        ansmap["pal"] = pal_ans
    if len(str(p2c_ans)) > 400: # over 400 --> truncate and string return
        print(f"truncating PAL result to 400 chars ({len(pal_ans)=})")
        p2c_ans = str(p2c_ans)[:400]
        ansmap["p2c"] = p2c_ans

    return ansmap, solmap, plan  # updated ones


def rims_complete_row(
    row: dict,
    temperature: float,
    n: int,
    backbone: str,
    seed: int,
    dataset_type: Literal["gsm", "ocw", "math"],
    prompt_f: str,
    # turn_based: bool,
):
    # try:
    if dataset_type == "ocw":
        question = row["problem"]
    else:
        question = row["question"]

    # individual method inference: this will check if row already has individual method inferred, and if done, keep those to use.
    ansmap, solmap, _plan = indiv_inference(
        row,
        num_methods=3,
        temperature=temperature,
        n=n,
        backbone=backbone,
        seed=seed,
        dataset_type=dataset_type,
    )
    row["ansmap"] = ansmap
    row["solmap"] = solmap
    row["plan"] = _plan

    # is there majority answer? in ansmap?
    majority_ans = get_concordant_answer(
        list(ansmap.values()), ensure_unanimity=False, dataset_type=dataset_type
    )

    # do rims
    if majority_ans is None:  # problems are not done properly.
        # here it had eval indiv_methods
        (
            eval_friendly_d,
            __,
            raw_query_out,
            query_msg,
            ___,
        ) = query_rims_inference(
            question,
            prompt_f,
            backbone=backbone,
            temperature=temperature,
            # turn_based=turn_based,
        )

        eval_friendly_d.update(
            {"raw_query_out": raw_query_out, "query_msg": query_msg}
        )
        row[
            "selection_or_rims"
        ] = eval_friendly_d  # this contains all we need depicted above
        row["majority_ans"] = eval_friendly_d["good_ans"]
    else:
        row["selection_or_rims"] = {"majority_vote": True}
        row["majority_ans"] = majority_ans
    row["prompt_file"] = str(prompt_f)
    # if turn_based:
    #     row["prompt_file"] += "_turn_based"
    row["inference_mode"] = "rims"
    # except Exception as e:
    #     print(e)
    #     print(f"error occured at {row['index']}")
    #     row["selection_or_rims"] = {"error": True, "exception": str(e)}
    #     row["majority_ans"] = None
    #     row["prompt_file"] = str(prompt_f)
    #     row["inference_mode"] = "rims"
    return row


def rims_inference(
    prompt_f: str = "prompt_construction_src/newer_prompts_3/rims_gsm_best.txt",
    gsm_jslf: str = "", # already baseline-inferred jsonlines file only!
    dataset_type: Literal[
        "gsm", "ocw", "math"
    ] = "gsm",  # affects get_concordant_answer
    backbone: str = "chatgpt0613long", # see llm_query_utils.backbone2model
    # turn_based: bool = False, # if True, convert the prompt into turn-based format and proceeds with it. 
    
    # llm options
    temperature: float = 0.0,
    n: int = 1,  # later for self-consistency
    seed: int = 777,
    start_idx: int = 0,
    err_idxs_f: str ="",
    
    # dev option
    dbg: bool = False,
):
    assert prompt_f, f"need to specify {prompt_f=}"
    assert gsm_jslf, f"need to specify {gsm_jslf=}"

    if n > 1:
        raise NotImplementedError(
            "n>1 will serve as a self-consistency parameter, not implemented yet"
        )

    
    # baseline `outdir` was like below
    # outdir = Path("outputs") / f"{Path(gsm_jslf).stem}_dt.{dataset_type}" / backbone / Path(prompt_f).stem
    # rims `outdir` below shares backbone
    outdir = Path(gsm_jslf).parent.parent/Path(prompt_f).stem
    
    # sanity check for the directory hierarchy
    assert Path(gsm_jslf).parent.parent.name == backbone, \
        f"inferred backbone differs with the current:\n \
            inferred: {Path(gsm_jslf).parent.parent=} != current: {backbone=}"
    assert Path(gsm_jslf).parent.parent.parent.name.endswith(dataset_type), \
        f"inferred dataset_type differs with the current:\n \
            inferred: {Path(gsm_jslf).parent.parent.parent.name=} != current: {dataset_type=}"

    if not outdir.exists():
        outdir.mkdir(parents=True)
    
    dt_string = f"{datetime.now():%m_%d_%H_%M_%S}"
    outpath = outdir / f"{'dbg_' if dbg else ''}{dt_string}.jsonl"

    # load_gsm_dataset to infer on
    records = list(jsl.open(gsm_jslf))[start_idx:]
    if err_idxs_f:
        assert start_idx==0, "err_idxs_f is only supported when start_idx is 0"
        idxs = [int(i) for i in open(err_idxs_f).read().strip().split("\n")]
        records = [records[i] for i in idxs]
        outpath = str(outpath).replace(".jsonl", ".jsonl_leftovers")
        while Path(outpath).exists():
            outpath += "_"
        outpath = Path(outpath)
    
    # remove duplicative rows (by question)
    print("dropping possible duplicates")
    df = pd.DataFrame(records)
    print("initiailly ", len(df), "records")
    k = "question" if "question" in df.columns else "problem"
    assert k in df.columns, f"{k} not in df.columns, {df}"
    df = df.drop_duplicates(subset=k)
    records = df.to_dict(orient="records")
    print("after dropping duplicates, ", len(df), "records")

    print(f"writing to \n\t{outpath}\n\n\n\n")

    # data to dataframe
    df = pd.DataFrame(records)

    # resolve index problem (so that dataframe and record both have index column)
    if "index" not in df.columns or (df["index"].isna().any()):
        df["index"] = range(len(df))
    df = df.set_index("index", drop=False) # df.index == df["index"] and "index" will appear in jsl

    # pick conflict only records to efficiently infer, keeping its order intact
    nonconflict_mask = df.selection_or_rims.apply(
        lambda d: d["majority_vote"] if "majority_vote" in d.keys() else False
    ) # if "majority_vote" field exists and is True, it's nonconflict
    
    to_process_df = df[~nonconflict_mask] # get conflicts only
    to_process_df.majority_ans = None  # clean up selection results from previous inference: to avoid contamination!
    to_process_df = to_process_df.drop(columns=["selection_or_rims"]) # clean up majority_ans

    # cleaned up!
    records_cleansed = to_process_df.to_dict(orient="records")

    _func = partial(
        rims_complete_row,
        temperature=temperature,
        n=n,
        backbone=backbone,
        seed=seed,
        dataset_type=dataset_type,
        prompt_f=prompt_f,
    )

    if dbg:
        for row in tqdm(records_cleansed):
            row = _func(row)  # updates rows in records_cleansed
        records_done = records_cleansed
    else:
        records_done = pqdm(records, _func, n_jobs=4) # to avoid BrokenPipe, keep n_jobs<=4 (tested on Mac M1)


    # nonconflict and processed conflict set of records remerged w/o index change
    df_done = pd.DataFrame(records_done)
    df_done = df_done.set_index(
        "index", drop=False
    )  
    df.loc[df_done.index] = df_done # updating only selection-done rows in the original df
    records_done = df.to_dict(orient="records")


    with jsl.open(outpath, "w") as writer, \
            open(f"{outpath}.errors", "w") as writer_err:
        for i, row in enumerate(records_done):
            try:
                if "index" in row.keys():
                    if row["index"] is None:
                        del row["index"]
                writer.write(row)
            except Exception as e:
                writer_err.write(str(records[i]) + "\n")
                writer_err.write(str(e) + "\n")
                print(e)
                print(f"{outpath}.errors")

    return


def baseline_complete_row(
    row: dict,
    temperature: float,
    n: int,
    backbone: str, 
    seed: int,
    prompt_f: str,
    dataset_type: Literal["gsm", "ocw", "math"],
    num_methods: int=3,
):
    try:
        if dataset_type == "ocw":
            question = row["problem"]
        else:
            question = row["question"]

        # individual method inference: this will check if row already has individual method inferred, and if done, keep those to use.
        ansmap, solmap, _plan = indiv_inference(
            row,
            num_methods=3,
            temperature=temperature,
            n=n,
            backbone=backbone,
            seed=seed,
            dataset_type = dataset_type,
        )

        row["ansmap"] = ansmap
        row["solmap"] = solmap
        row["plan"] = _plan

        # is there majority answer? in ansmap? (2,2,1 --> 2 is majority, can assert hard condition such as requiring unanimous votes)
        majority_ans = get_concordant_answer(
            list(ansmap.values()), ensure_unanimity=False, dataset_type=dataset_type
        )

        # ensure solmap is Dict[str, str]
        for k,v in solmap.items():
            if not isinstance(v, str):
                if isinstance(v[0],str):
                    solmap[k] = v[0]
                else:
                    raise ValueError(f"sth's going wrong with {solmap=}")

        if majority_ans is None:  # do selection
            chosen_method, selection_str, _ = query_selection(
                question,
                backbone=backbone,
                cot_solution=solmap["cot"],
                pal_solution=solmap["pal"],
                p2c_plan_code_solution=solmap["p2c"],
                dataset_type=dataset_type,
            )
            if chosen_method is not None: 
                row["selection_or_rims"] = {
                    "good_method": chosen_method,
                    "good_answer": ansmap[chosen_method],
                    "good_solution": solmap[chosen_method],
                    "selection_str": selection_str,
                    "dataset_type": dataset_type,
                }
            else:
                row['selection_or_rims'] = {
                    "good_method": None,
                    "good_answer": None,
                    "good_solution": None,
                    "selection_str": selection_str,
                    "dataset_type": dataset_type,
                }
            row["majority_ans"] = row['selection_or_rims']['good_answer']
        else:
            row["selection_or_rims"] = {"majority_vote": True}
            row["majority_ans"] = majority_ans
        row["prompt_file"] = prompt_f
        row["inference_mode"] = f"baseline {num_methods} methods"
    except Exception as e:
        print(e)
        row["selection_or_rims"] = {"error": True, "exception": str(e)}
        
    return row


def baseline_inference(
    prompt_f: str = "prompt_construction_src/newer_prompts_3/model_selection_prompts.yaml",
    gsm_jslf: str = "",
    dataset_type: Literal[
        "gsm", "ocw", "math"
    ] = "gsm",  # affects get_concordant_answer
    num_methods: int = 3,  # number of methods (3-> cot pal p2c / 2-> cot pal )
    start_idx: int = 0,
    err_idxs_f: str ="", # only when start_idx == 0
    
    # llm options
    temperature: float = 0.0,
    n: int = 1,  # later for self-consistency
    backbone: str = "chatgpt0613long",  
    seed: int = 777,
    
    # dev option
    dbg: bool = False,

):
    assert gsm_jslf, f"need to specify {gsm_jslf=}"
    if prompt_f != "prompt_construction_src/newer_prompts_3/model_selection_prompts.yaml":
        raise ValueError(f"llm_query_utils.get_select_prompt2() reads prompt_construction_src/newer_prompts_3/model_selection_prompts.yaml, not the {prompt_f=} provided!")

    if n > 1:
        raise NotImplementedError(
            "n>1 will serve as a self-consistency parameter, not implemented yet"
        )

    # load_gsm_dataset to infer on
    records = list(jsl.open(gsm_jslf))[start_idx:]

    # output directory for the inference results:
    # outpath = outputs/{dataset_stem}/{prompt_stem}/{now}.jsonl
    outdir = Path("outputs") / f"{Path(gsm_jslf).stem}_dt.{dataset_type}" / backbone / Path(prompt_f).stem

    if not outdir.exists():
        outdir.mkdir(parents=True)
    
    dt_string = f"{datetime.now():%m_%d_%H_%M_%S}"
    outpath = outdir / f"{'dbg_' if dbg else ''}{dt_string}.jsonl"

    # handle only error indexes, discard otherwise
    if Path(err_idxs_f).exists() and err_idxs_f:
        assert start_idx==0, "err_idxs_f is only supported when start_idx is 0"
        idxs = [int(i) for i in open(err_idxs_f).read().strip().split("\n")]
        records = [records[i] for i in idxs]
        outpath = str(outpath).replace(".jsonl", ".jsonl_leftovers")
        while Path(outpath).exists():
            outpath += "_"
        outpath = Path(outpath)

    # remove duplicative rows (by question)
    print("dropping possible duplicates")
    df = pd.DataFrame(records)
    print("initiailly ", len(df), "records")
    k = "question" if "question" in df.columns else "problem"
    assert k in df.columns
    df = df.drop_duplicates(subset=k)
    records = df.to_dict(orient="records")
    print("after dropping duplicates, ", len(df), "records")

    _func = partial(
        baseline_complete_row,
        temperature=temperature,
        n=n,
        backbone=backbone,
        seed=seed,
        dataset_type=dataset_type,
        prompt_f=prompt_f,
        num_methods=num_methods,
    )

    print(f"writing to \n\t{outpath}\n\n\n\n")

    if dbg:
        for row in tqdm(records):
            out = _func(row)
            row = out
        records_done = records
    else:
        records_done = pqdm(records, _func, n_jobs=5)
    
    with jsl.open(outpath, "w") as writer, \
            open(f"{outpath}.errors", "w") as writer_err, \
            open(f"{outpath}.error_idxs", "w") as writer_err_idx:
        for i, row in enumerate(records_done):
            try:
                if "index" in row.keys():
                    if row["index"] is None:
                        del row["index"]
                writer.write(row)
            except Exception as e:
                writer_err.write(str(records[i]) + "\n")
                writer_err.write(str(e) + "\n")
                writer_err_idx.write(str(i)+"\n")
                print(e)
                print(f"{outpath}.errors")
                print(f"{outpath}.error_idx")

        return

if __name__ == "__main__":
    Fire()
