import time
from openai import OpenAI

class BaseAgent():
    def __init__(self, name: str, description: str = ""):
        # self.client = OpenAI(api_key=openai_api_key)
        # self.client_type = "OpenAI"
        self.message_history = []
        self.name = name
        self.description = description

    def process_message(self, query: str):
        """Process the input query and return a result"""
        pass

    def timed(self, query: str):
        """Time the agent processing"""
        start = time.perf_counter()
        result = self.process_message(query)
        duration = time.perf_counter() - start

        return result, duration