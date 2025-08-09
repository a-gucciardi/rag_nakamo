from openai import OpenAI
import json

class OrchestratorAgent():
    """
    The orchestrator decides which agents to use (currently RAG or summary).
    Uses LLM function calling to determine the appropriate action plan.
    """

    def __init__(self, client="OpenAI", openai_api_key=None, hf_model="google/gemma-3-4b-it", ollama_model="llama3.1:8b"):
        self.client_type = client

        if client == "OpenAI":
            self.client = OpenAI(api_key=openai_api_key)

        elif client == "HF":
            from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
            import torch, os
            os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                hf_model, 
                device_map="auto", 
                torch_dtype=torch.bfloat16,
                attn_implementation="eager"
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)

        elif client == "Ollama":
            import ollama
            self.ollama_model = ollama_model
            self.ollama = ollama

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

        system_prompt = """
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

        if self.client_type == "OpenAI":
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
            if response.choices[0].message.function_call:
                function_call = response.choices[0].message.function_call
                function_name = function_call.name
                function_args = json.loads(function_call.arguments)

                action_plan = {
                    "function": function_name,
                    "arguments": function_args,
                    "original_question": message
                }
                return action_plan

        elif self.client_type == "HF":
            import torch
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Regulatory question: {message}"}
                ]
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(**inputs, max_new_tokens=50, do_sample=False)

            response_text = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            print(f"Response from HuggingFace model: {response_text}")

            if "rag agent" in response_text.lower():
                return {
                    "function": "use_rag_agent",
                    "arguments": {"query": message},
                    "original_question": message
                }
            else:
                return {
                    "function": "generate_response", 
                    "arguments": {"question": message, "retrieved_info": "", "sources": []},
                    "original_question": message
                }

        elif self.client_type == "Ollama":
            full_prompt = f"""{system_prompt}
            Regulatory question: {message}
            Which agent should be used first? Respond with "use_rag_agent" or "generate_response" only.
            """
            response = self.ollama.chat(model=self.ollama_model, messages=[
                {"role": "user", "content": full_prompt}
            ])
            reply = response['message']['content'].strip().lower()
            print(f"Response from Ollama: {reply}")

            if "rag" in reply:
                return {
                    "function": "use_rag_agent",
                    "arguments": {"query": message},
                    "original_question": message
                }
            else:
                return {
                    "function": "generate_response",
                    "arguments": {"question": message, "retrieved_info": "", "sources": []},
                    "original_question": message
                }

        return None
