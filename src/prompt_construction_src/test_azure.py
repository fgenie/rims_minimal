from openai import AzureOpenAI
import os 

print(os.getenv("AZURE_OPENAI_ENDPOINT"))
print(os.getenv("AZURE_OPENAI_API_KEY"))
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-12-01-preview",
)

'''
GPT-35 : gpt-3.5-turbo-0613   / 4k context window, 
GPT4-1106: gpt-4-1106-preview # gpt-4-turbo 1/3 cost / 128,000 context window, 4k output 
''' 


kwargs = dict(
        temperature=0.7, 
        model='GPT-35', # 'GPT4-1106' 
        seed=777,
        n=5,
        messages = [{"role": "user", "content": "are you working?"}]
    )

response = client.chat.completions.create(
    **kwargs
)

print(response)
print()