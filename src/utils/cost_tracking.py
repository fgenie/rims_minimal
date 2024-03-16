from functools import update_wrapper
from typing import Dict, Any

class CountTokens:
    """
    # llm_query_utils.py
    @CountTokens
    def query_cot():
        return

    # main.py
    ...
    ..
    .
    
    query_cot.print_summary()
        
    
    will initialize a CountTokens object with the query_cot function as the `self.func` attribute, but letting function profile as `query_cot`
    
    This decorator will count the total number of tokens used by the function 
    """
    def __init__(self, func):
        self.func = func
        self.n_called = 0
        # for cost tracking
        self.total_toks_in = 0
        self.total_toks_out = 0
        # for context length estimation 
        self.max_toks_in = 0
        self.max_toks_out = 0 
        update_wrapper(self, func)  # Update the wrapper function (self) to look like func

    def tok_info_from_query_funcs(
            self,
            query_returns:Any=None
            )->Dict[str, Any]:
        """
        from `query_returns` of query functions (query_cot|pal|plancode|selection...etc), get the total_number of the 
        """
        funcname = self.func.__name__
        tok_info:Dict[str, int] = dict.fromkeys(["toks_in", "toks_out"], 0)
        
        if funcname in "query_cot query_pal _query":
            query_returns
            raise NotImplementedError("Not implemented yet")
        elif funcname in "query_selection":
            raise NotImplementedError("Not implemented yet")
        elif funcname in "query_rims_inference":
            raise NotImplementedError("Not implemented yet")
        
        return tok_info

    def __call__(self, *args, **kwargs):
        # Call the original function
        results = self.func(*args, **kwargs)
        
        inout_tokens_d:Dict[str, int] = self.tok_info_from_query_funcs(results)
        toks_in, toks_out = inout_tokens_d["toks_in"], inout_tokens_d["toks_out"]

        # update counts 
        self.n_called += 1
        self.max_toks_in = max(self.max_token_in, toks_in)
        self.max_toks_out = max(self.max_token_out, toks_out)
        self.total_toks_in += toks_in
        self.total_toks_out += toks_out
        
        return results # return the self.func.__call__() results

    def print_summary(self):
        print(f"Function: {self.func.__name__}")
        print(f"Max tokens in: {self.max_toks_in}")
        print(f"Max tokens out: {self.max_toks_out}")
        print(f"Total tokens in: {self.total_toks_in}")
        print(f"Total tokens out: {self.total_toks_out}")
        print(f"Number of calls: {self.n_called}")
    


def tokens2usd(toks_in: int=0, toks_out: int=0, model: str="") -> float:
    """
    # Example usage
    cost = tokens2usd(toks_in=500_000, toks_out=500_000, model="gpt-3.5-turbo-1106")
    print(f"Cost: ${cost:.4f}")
    """
    # Define the cost per 1,000,000 tokens for each model type for input and output
    pricing = {
        "gpt-3.5-turbo-1106": {"input": 1.00, "output": 2.00},
        "gpt-3.5-turbo-0613": {"input": 1.50, "output": 2.00},
        "gpt-3.5-turbo-16k-0613": {"input": 3.00, "output": 4.00},
        "gpt-3.5-turbo-0301": {"input": 1.50, "output": 2.00},
        "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
        "gpt-3.5-turbo-instruct": {"input": 1.50, "output": 2.00},
        "gpt-4-1106-preview": {"input": 10.00, "output": 30.00},
        "gpt-4-0125-preview": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-4-32k": {"input": 60.00, "output": 120.00},
    }

    # Check if the provided model is in the pricing dictionary
    if model not in pricing:
        raise ValueError(f"Unknown model type: {model}")

    # Calculate the cost in USD for input and output tokens separately
    cost_in_usd = (toks_in / 1_000_000) * pricing[model]["input"] + (toks_out / 1_000_000) * pricing[model]["output"]

    return cost_in_usd
