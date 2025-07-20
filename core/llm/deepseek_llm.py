from langchain_deepseek import ChatDeepSeek
from dotenv import load_dotenv
import os
from typing import List

load_dotenv()

api_key = os.getenv("Deepseek_api_key") 

class DeepSeekLLM:
    def __init__(self, model_name: str = "deepseek-chat", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key

    def get_llm(self):
        return ChatDeepSeek(model_name=self.model_name, api_key=self.api_key)

    # def get_query_prompt(self, question: str) -> List[str]:
    #     self.template = [(question)]
    #     return self.template
    
    def post_request(self, prompt: str) -> List[str]:
        llm = self.get_llm()
        response = llm.invoke(prompt)
        return response.content
    
if __name__ == "__main__":
    deepseek_llm = DeepSeekLLM(api_key=api_key)
    messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
    ]
    # prompt = deepseek_llm.get_query_prompt("What is the capital of France?")
    response = deepseek_llm.post_request(messages)
    print(response)