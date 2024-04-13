import os

from openai import AzureOpenAI

# print(os.getenv("OLD_AZURE_OPENAI_ENDPOINT")) # https://laba.openai.azure.com
# print(os.getenv("OLD_AZURE_OPENAI_API_KEY")) # key

"""
GPT-35 : gpt-3.5-turbo-0613   / 4k context window,
GPT4-1106: gpt-4-1106-preview # gpt-4-turbo 1/3 cost / 128,000 context window, 4k output
"""


useold = True


if useold:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("OLD_AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("OLD_AZURE_OPENAI_API_KEY"),
        api_version="2023-07-01-preview",
    )

    models = [
        # "GPT-35",
        "GPT4-1106"
    ]
else:
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2023-03-01-preview",
    )

    models = [
        "laba-gpt-35-turbo-0125",
        "laba-gpt-35-turbo-1106",
        "laba-gpt-35-turbo-16k-0613",
    ]


for mod in models:
    response = client.chat.completions.create(
        temperature=0.7,
        model=mod,  # 'laba-gpt-35-turbo-0125', #or "GPT-35",
        seed=777,
        n=2,
        messages=[{"role": "user", "content": "are you working?"}],
    )

    print(mod)
    print(response.model)  # ChatCompletion object
    print(response.usage.completion_tokens)  # regardless of n=1 or n>1
    print(response.usage.prompt_tokens)
    print()
