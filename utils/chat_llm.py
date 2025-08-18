import dotenv
import os
import requests
from openai import AzureOpenAI, OpenAI
from guardrails import Guard
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import json
from .logger import configure_logger

dotenv.load_dotenv()

logger = configure_logger()


class ChatLLM:
    """
    chat llm class
    """

    def __init__(self, model_name: str, cache_path: str = None):
        """
        init chat llm
        Args:
            model_name: model name
        """
        self.model_name = model_name
        self.openai_model_name = os.getenv("OPENAI_MODEL_NAME")

        # cache llm's response
        self.cache = {}
        self.cache_path = cache_path or ".cache/llm.jsonl"
        if not os.path.exists(self.cache_path):
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                f.write("")
        self.load_cache()

    def delete_last_cache_item(self):
        with open(self.cache_path, "r+") as f:  # 读写模式打开文件
            # 将指针移动到文件末尾前2个字符的位置（防止最后一行没有换行符）
            f.seek(0, os.SEEK_END)
            pos = f.tell() - 2  # 从文件末尾回退两个字符
            # 反向查找最后一个换行符
            while pos > 0 and f.read(1) != "\n":
                pos -= 1
                f.seek(pos, os.SEEK_SET)
            # 截断文件到找到的位置
            f.truncate(pos + 1 if pos > 0 else 0)

    def save_cache(self, cache_key, response):
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"cache_key": cache_key, "response": response}, ensure_ascii=False
                )
                + "\n"
            )

    def load_cache(self):
        self.cache = {}
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.cache[data["cache_key"]] = data["response"]

    def add_cache(self, cache_key, response):
        if cache_key in self.cache.keys():
            return
        self.cache[cache_key] = response
        self.save_cache(cache_key, response)

    def search_cache(self, cache_key):
        return self.cache.get(cache_key, None)

    def chat_completion(self, messages, use_cache: bool = True, **kwargs):
        """
        chat completion, choose chat completion method according to model_name
        Args:
            messages: list of messages
            use_cache: whether to use cache
            **kwargs: other arguments
        Returns:
            response: response from azure openai
        """
        # 拼接参数
        cache_key = str(self.model_name) + str(messages) + str(kwargs)
        if use_cache:
            response = self.search_cache(cache_key)
            if response is not None:
                return response

        if self.model_name in os.getenv("QWEN_MODEL_NAME"):
            response = self.__qwen_chat_completion(messages)
        else:
            raise ValueError(f"{self.model_name} is not supported")
        
        if use_cache:
            self.add_cache(cache_key, response)
        return response

    def __azure_chat_completion(self, messages):
        """
        chat completion with azure openai
        Args:
            messages: list of messages
        Returns:
            response: response from azure openai
        """
        openai_client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_ENDPOINT"),
            api_key=os.getenv("AZURE_API_KEY"),
            api_version=os.getenv("AZURE_API_VERSION"),
        )
        response = openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content

    def __deepseek_chat_completion(self, messages):
        """
        chat completion with deepseek
        Args:
            messages: list of messages
        Returns:
            response: response from deepseek
        """
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        response = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model_name,
                "messages": messages,
                "response_format": {"type": "text"}
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    
    def __llama_chat_completion(self, messages):
        """
        chat completion with llama
        Args:
            messages: list of messages
        Returns:
            response: response from llama
        """
        base_url = os.getenv("LLAMA_BASE_URL")
        api_key = os.getenv("LLAMA_API_KEY")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        response = client.chat.completions.create(
            messages=messages,
            model=self.model_name,
        )
        return response.choices[0].message.content

    def __qwen_chat_completion(self, messages):
        """
        chat completion with qwen
        Args:
            messages: list of messages
        Returns:
            response: response from qwen
        """
        base_url = os.getenv("QWEN_BASE_URL")
        api_key = os.getenv("QWEN_API_KEY")
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return response.choices[0].message.content
    
    

@retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(2))
def chat_llm(
    *, messages: list | str, model_name: str = None, verbose: bool = False, use_cache: bool = True, **kwargs
):
    """
    chat llm
    Args:
        messages: messages
        model_name: model name, if None, use the model name in environment variable
        verbose: whether to log the messages and response
        **kwargs: other arguments
    Returns:
        response: response from chat completion
    """
    model_name = model_name or os.getenv("DEPLOYMENT_NAME")
    chat_llm = ChatLLM(model_name)

    if type(messages) == str:
        messages = [
            {"role": "system", "content": "You are a assistant."},
            {"role": "user", "content": messages},
        ]
    try:
        response = chat_llm.chat_completion(messages=messages, use_cache=use_cache, **kwargs)
        if verbose:
            logger.info(
                repr(f"{model_name} chat. Messages: {messages} Response: {response}")
            )
        return response
    except Exception as e:
        logger.error(f"API Error: {e}")
        return ""


def chat_llm_guard(
    messages: list | str,
    pydantic_model: BaseModel,
    model_name: str = None,
    retry_times: int = 2,
    verbose: bool = False,
    **kwargs,
):
    """
    chat llm with guard
    Args:
        messages: messages
        pydantic_model: structure of the response
        model_name: model name, if None, use the DEPLOYMENT_NAME in environment variable
        retry_times: retry times
        verbose: whether to log the messages and response
        **kwargs: other arguments
    Returns:
        response: response from chat completion
    """
    try:
        model_name = model_name or os.getenv("DEPLOYMENT_NAME")

        if type(messages) == str:
            messages = [
                {"role": "system", "content": "You are a assistant."},
                {"role": "user", "content": messages},
            ]
        for _ in range(retry_times):
            guard = Guard.for_pydantic(pydantic_model)
            response = guard(
                chat_llm,
                messages=messages,
                model_name=model_name,
                verbose=verbose,
                **kwargs,
            )
            if response.validation_passed:
                return response.validated_output
            else:
                # llm response就会保存到cache，需要删掉该cache，否则重复访问仍是错的。
                chat_llm_instance = ChatLLM(model_name)
                chat_llm_instance.delete_last_cache_item()
        logger.error(
            f"Failed to validate response after {retry_times} times.\n {model_name} {messages}"
        )
        return ""

    except Exception as e:
        logger.error(f"API Error: {e}")
        # llm response就会保存到cache，需要删掉该cache，否则重复访问仍是错的。
        chat_llm_instance = ChatLLM(model_name)
        chat_llm_instance.delete_last_cache_item()
        return ""


if __name__ == "__main__":
    model_list = ["deepseek-ai/DeepSeek-V2.5", "azureai", "qwen2.5-7b-instruct-1m"]
    for model in model_list:
        response = chat_llm(messages="Can you generate a list of 10 things that are not food?", model_name=model)
        print(response)

    # guard
    class Answer(BaseModel):
        role: str
        content: str

    response = chat_llm_guard(
        model_name="gpt-4o",
        messages="Can you generate a list of 10 things that are not food?",
        pydantic_model=Answer,
        retry_times=2,
    )

    print(response)
