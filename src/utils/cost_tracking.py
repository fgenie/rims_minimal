from functools import update_wrapper
from typing import Dict, Any

class CountTokens:
    """
    @CountTokens
    def query_cot():
        return

    will initialize a CountTokens object with the query_cot function as the func attribute, but letting function profile appear as `query_cot`
    
    will count and record the total number of tokens used by the function 
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
    
