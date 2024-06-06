"""
running
cot
pal
p2c

for a designated dataset



resources:

src/run_inference.py:indiv_inference()
src/utils/llm_query_utils.py
    src/utils/math_prompt.py # gsm cot/pal prompts
    src/utils/plancode_util_v2.py # plan2code prompts

"""

from query import CoTQueryObject, PALQueryObject, P2CQueryObject


async def indiv_query(
    row: dict = None,
    num_methods: int = 3,
    temperature: float = 0.0,
    p2c_plan_temperature: float = 0.0,
    n: int = 1,
    seed: int = 777,
    backbone: str = "chatgpt0613long",
    dataset_type: Literal["gsm", "ocw", "math"] = "",
    only_retrieve: bool = False,  # if true, when undone thing found, throws error
):
    """
    inference each method and return indiv results
    if there are already existing results, use them.


    return:
        solmap : {cot: pal: p2c:}
        ansmap : {cot: pal: p2c:} (solution executed)
    """

    if dataset_type == "ocw":
        question = row["problem"]
    else:
        question = row["question"]
        
    missing_methods = ["cot", "pal", "p2c"]

    if only_retrieve:
        raise ValueError(
            f"no existing results found while {only_retrieve=}\n\n{missing_methods=}\n{ansmap=}"
        )
    
    query_objects = {
        "cot": CoTQueryObject,
        "pal": PALQueryObject,
        "p2c": P2CQueryObject,
    }

    max_tokens = {
        "cot": {
            "gsm": 400,
            "ocw": 850,
            "math": 950,
        },
        "pal": {
            "gsm": 350,
            "ocw": 500,
            "math": 400,
        },
        "p2c": {
            "gsm": 1024,
            "ocw": 1024,
            "math": 1024,
        }
    }

    return_data = {}
    
    for method in missing_methods:
        # function prepare variables
        init_param = {
            'dataset_type': dataset_type
        }
        if method == "p2c":
            init_param["plan_temperature"] = p2c_plan_temperature

        query_params = {
            "question": question,
            "temperature": temperature,
            "backbone": backbone,
            "n": n,
            "seed": seed,
            "max_tokens": max_tokens[method][dataset_type]
            stop="\n\n\n"
        }
        
        if method == "p2c":
            query_params["stop"] = "Question: "

        query_obj = query_objects[method](**init_param)
        contents, query_message, resp = await query_obj.async_query(**query_params)
        return_data[method] = {
            "contents": contents,
            "query_message": query_message,
            "resp": resp
        }
    return return_data