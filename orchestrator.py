import json
from openai import OpenAI

class OrchestratorAgent():
    """
    The orchestrator decides which agents to use (currently RAG or summary).
    Uses LLM function calling to determine the appropriate action plan.
    """

    def __init__(self, openai_api_key):
        self.client = OpenAI(api_key=openai_api_key)
        self.available_functions = {
            "use_rag_agent": {
                "name": "use_rag_agent",
                "description": "Search regulatory documents for relevant information using the RAG agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant regulatory information"
                        },
                        "focus_areas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific regulatory areas to focus on (e.g., 'FDA', 'WHO', 'software', 'design controls')"
                        }
                    },
                    "required": ["query"]
                }
            },
            "generate_response": {
                "name": "generate_response",
                "description": "Generate the final structured response using the Response agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The original regulatory question"
                        },
                        "retrieved_info": {
                            "type": "string",
                            "description": "The information retrieved from regulatory documents"
                        },
                        "sources": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of source documents referenced"
                        }
                    },
                    "required": ["question", "retrieved_info", "sources"]
                }
            }
        }
        self.message_history = []

    def process_message(self, message):
        """
        Process the input question and orchestrate the response workflow
        """
        self.message_history.append(message)

        # Orchestrator system prompt
        system_prompt = """
        You are an orchestrator for a medtech regulatory assistant system. 
        Your job is to analyze regulatory questions and decide which agents to use and in what order.

        Available agents: 
        1. RAG Agent: Searches regulatory PDFs for relevant information
        2. Response Agent: Creates structured final answers with citations

        You should:
        1. First use the RAG agent to search for relevant information
        2. Then use the Response agent to format the final answer

        Analyze the user's question and determine the appropriate workflow."""


        # allows to use the function calls
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Regulatory question: {message}"}
            ],
            functions=[func for func in self.available_functions.values()],
            function_call="auto",
            temperature=0
        )

        # print(response)

        # create action plan ?
        if response.choices[0].message.function_call:
            print("Orchestrator received function call and decided to use :", response.choices[0].message.function_call.name)
            function_call = response.choices[0].message.function_call
            function_name = function_call.name
            function_args = json.loads(function_call.arguments)

            action_plan = {
                "function": function_name,
                "arguments": function_args,
                "original_question": message
            }

            print(f"Action plan created: {action_plan}")

            return action_plan
