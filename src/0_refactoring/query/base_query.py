import os
from openai import OpenAI, AsyncOpenAI


base_url = os.environ.get("OPENAI_API_BASE", "http://localhost:49999/v1")
api_key = os.environ.get("OPENAI_API_KEY", "no_need")
timeout = int(os.environ.get("OPENAI_TIMEOUT", 120))
max_retries = int(os.environ.get("OPENAI_MAX_RETRY", 4))

async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )


class BaseQueryObject:
    async def async_query(
        self,
        question: str,
        temperature: float = 0.0,
        backbone: str = "chatgpt",
        n: int = 1,
        seed: int = 777,
        max_tokens: int = 2048,
        stop="\n\n\n",
    ):
        
        meta = {'method_obj': self.__class__.__name__}
        
        prepare_query_task = self.prepare_query(question, backbone=backbone, **{
            "temperature": temperature,
            "n": n,
            "seed": seed,
            "max_tokens": max_tokens,
            "stop": stop
        }) 
        query_message = await prepare_query_task
        is_error, error_msg = self.query_error_msg(query_message)
        if is_error:
            return error_msg
        
        model_name = self.backbone2model(backbone)
    
        call_llm_task = self.async_query_to_llm(
            model=model_name,
            max_tokens=max_tokens,
            stop=stop,
            messages=query_message,
            temperature=temperature,
            top_p=1.0,
            seed=seed,
            n=n,
        )
        resp = await call_llm_task
        contents = self.get_contents(resp)
        return contents, query_message, resp, meta

    async def prepare_query(
        self,
        question: str,
        backbone: str,
        **kwargs
    ):
        raise NotImplementedError()

    def query_error_msg(self, query_message):
        raise NotImplementedError()

    @staticmethod
    async def async_query_to_llm(**chat_params):
        res = await async_client.chat.completions.create(**chat_params)
        return res

    @staticmethod
    def get_contents(resp):
        completions = [choice.message.content for choice in resp.choices]
        return completions

    @staticmethod
    def backbone2model(backbone: str) -> str:
        if backbone == "gpt4":
            model_name = "gpt-4"
        elif backbone == "gpt4turbo":  # or backbone == "GPT4-1106":
            # model_name = "gpt-4-1106-preview"
            model_name = "GPT4-1106"
        elif backbone == "chatgpt0613":  # or backbone == "GPT-35":
            model_name = "gpt-3.5-turbo-0613"
        elif backbone == "chatgpt0125":
            # model_name = "gpt-3.5-turbo-0125"
            model_name = "laba-gpt-35-turbo-0125"
        elif backbone == "chatgpt1106":
            # model_name = "gpt-3.5-turbo-1106"
            model_name = "laba-gpt-35-turbo-1106"
        elif backbone == "chatgpt0613long":
            # model_name = "gpt-3.5-turbo-16k-0613"
            model_name = "laba-gpt-35-turbo-16k-0613"
        else:
            model_name = backbone  # openLLM setting
        return model_name