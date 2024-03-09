from openai import AzureOpenAI
import os 

# print(os.getenv("OLD_AZURE_OPENAI_ENDPOINT")) # https://laba.openai.azure.com
# print(os.getenv("OLD_AZURE_OPENAI_API_KEY")) # key
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-03-01-preview", #"2023-07-01-preview",
)

'''
GPT-35 : gpt-3.5-turbo-0613   / 4k context window, 
GPT4-1106: gpt-4-1106-preview # gpt-4-turbo 1/3 cost / 128,000 context window, 4k output 


''' 

models = [
    "laba-gpt-35-turbo-0125",
"laba-gpt-35-turbo-1106",
"laba-gpt-35-turbo-16k-0613",
]

for mod in models: 

    response = client.chat.completions.create(
            temperature=0.7, 
            model=mod, # 'laba-gpt-35-turbo-0125', #or "GPT-35",
            seed=777,
            n=5,
            messages = [{"role": "user", "content": "are you working?"}]
    )

    print(mod)
    print(response.model) # ChatCompletion object
    print()