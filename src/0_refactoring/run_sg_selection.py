"""
running
simple-greedy selection
(chul feels comfortable with calling the following as as above: Automatic Model Selection Reasoning which is our baseline https://arxiv.org/abs/2305.14333)

** note: selection algorithm supposed to run only where majority vote failed to reach the consensus **

for a designated dataset


- [ ] no self-consistency (n=1)
- [ ] self-consistency (n>1)

**
according to the author's code, self consistency is implemented with n-iterative call (https://github.com/XuZhao0/Model-Selection-Reasoning/blob/8ee494276e958c3f88332d4be64f8a746395f11c/src/selection_math.py#L301),
where we replace this with client.chat.completions.create(n=n)



----
resources:
    src/utils/model_selection_prompts.yaml # simple-greedy prompts for MATH, ocw dataset (gsm in math_prompt.py)
    src/utils/math_prompts.py


"""
