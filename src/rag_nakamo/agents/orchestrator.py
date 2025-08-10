from rag_nakamo.agents.base import BaseAgent
from rag_nakamo.settings import get_settings
from openai import OpenAI
import logging, json

logger = logging.getLogger(__name__)

class OrchestratorAgent(BaseAgent):
    """Orchestrator Agent for managing the RAG process.
    This agent coordinates the flow of information and actions between different components of the RAG system.
    It processes action plans and delegates tasks to other agents as needed.
    """

    def __init__(self, name: str = "Orchestrator", description: str = "Orchestrates the RAG process"):
        super().__init__(name, description)
        self.settings = get_settings()
        self.model = self.settings.orchestrator_model
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.available_functions = {
            "use_rag_agent": {
                "name": "use_rag_agent",
                "description": "Search regulatory documents for relevant information using the RAG agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The input search query to find relevant regulatory information"
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
        self.system_prompt = """
        You are an orchestrator for a medtech regulatory assistant system. 
        Your job is to analyze regulatory questions and decide which agents to use and in what order.

        Available agents: 
        1. RAG Agent: Searches regulatory of both FDA and WHO PDFs for relevant information
        2. Response Agent: Creates structured final answers with citations

        You should:
        1. First use the RAG agent to search for relevant information in FDA and WHO regulatory documents.
        2. Then use the Response agent to format the final answer
        3. If the question is not related to regulations or FDA and WHO, return an error message

        Analyze the user's question and determine the appropriate workflow."""
        logger.info(f"Orchestrator initialized with model: {self.model}")
    
    def process_message(self, query: str):
        """Returns an action plan."""

        logger.info(f"Processing query: {query}")
        response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Regulatory question: {query}"}
                ],
                functions=[func for func in self.available_functions.values()],
                function_call="auto",
            )
        
        if response.choices[0].message.function_call:
            function_call = response.choices[0].message.function_call
            function_name = function_call.name
            function_args = json.loads(function_call.arguments)

            action_plan = {
                "function": function_name,
                "arguments": function_args,
                "original_question": query
            }
            return action_plan

    
