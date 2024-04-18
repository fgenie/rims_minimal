# evaluation script for self-consistency setting
# i.e. jsonlines file fields differ from n==1 case


# will check: majority_ans == GT answer
"""
simple greedy
            # update row: need to consider later it will be reused for rims inferencing.
            row["error"] = False
            row["error_msg"] = ""
            row["runnning_at"] = "baseline_complete_row"

            row["majority_ans"] = majority_ans
            row["idx2chosen_method"] = idx2chosen_method
            row["majvote_ans"] = majvote_ans
            row["candid_answers"] = candid_answers  # for debug use
            row["inference_mode"] = [
                "majority_vote" if majvote_ans is not None else "selection"
                for majvote_ans in majvote_ans
            ]
            row["dataset_type"] = dataset_type
            row["prompt_file"] = prompt_f
            row["temperatures"] = {
                "cot_temperature": 0.5,
                "pal_temperature": 0.8,
                "n": n,
            }


rims
            # update row
            row["error"] = False
            row["error_msg"] = ""
            row["running_at"] = "rims_complete_row"

            row["majority_ans"] = majority_ans
            row["idx2chosen_method"] = idx2chosen_method
            # row["majvote_ans"] = majvote_ans # not changed
            row["candid_answers"] = candid_answers
            row["inference_mode"] = ["majority_vote" if majvote_ans is not None else "rims" for majvote_ans in majvote_ans]
            row["dataset_type"] = dataset_type # for logging use... overwrite the dataset_type
            row["prompt_file"] = str(prompt_f)
            row["temperatures"].update(
                {"rims_temperature": temperature, "n": n, "n_adj": n_adj}
            )
            row["rims_query_results"] = eval_friendly_d_ # aggregated rims results
"""

# according to the fields above


# for math and ocw
#   majority_ans: List (length 1 or 2)
# if 2, check if any of those are correct (considers correct if any of them are correct)
