import re, json
import logging, time
from typing import List, Dict, Any
from openai import OpenAI
from rag_nakamo.settings import get_settings
from rag_nakamo.agents.base import BaseAgent

logger = logging.getLogger(__name__)

class ResponseAgent(BaseAgent):
    """ Final answer formatter and validator for regulatory content. """

    def __init__( self, name: str, description: str, enable_regulatory_formatting: bool = True):
        super().__init__(name, description)
        settings = get_settings()
        self.model = settings.response_model
        self.validation_model = getattr(settings, "validation_model", self.model)
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.enable_regulatory_formatting = enable_regulatory_formatting
        logger.info(f"Responser initialized with model: {self.model}")

    def process_message(self, query, content):
        """ Process and format the final answer.
            content : output of RAGAgent
        """
        # logger.info(f"Processing message with ResponseAgent: {self.name}")
        # Format the answer using LLM if available and enabled
        formatted_answer = self._format_answer_with_llm(query, content)
        return formatted_answer

    def _format_answer_with_llm(self, question: str, content: List[Dict[str, Any]]) -> str:
        """Format the answer using LLM with regulatory expert prompt."""

        regulatory_prompt = """You are a regulatory expert assistant. 
        Your task is to provide comprehensive, accurate answers to a given regulatory QUESTION about medical devices based on the provided regulatory documents content.

        IMPORTANT GUIDELINES:
        1. Base your answer ONLY on the provided regulatory documents snippets
        2. Provide a structured response with clear sections
        3. Include specific citations for each major point
        4. If the documents don't contain enough information, clearly state this
        5. Use professional, technical language appropriate for regulatory context
        6. Highlight key requirements, processes, or standards mentioned
        7. Compare FDA vs WHO approaches when relevant

        RESPONSE STRUCTURE:
        - ## Executive Summary (brief overview) and key requirements (if applicable)
        - ## Detailed Analysis (main content with citations from the first answer)
        - ## Sources (list all referenced documents, with pages)

        Use citation format: [Source Name, Page X] after each major point and at the end Sources."""

        user_prompt = """
        QUESTION: {question}

        CURRENT ANSWER CONTENT TO CREATE A FINAL RESPONSE FROM: {content}

        Please reformat the current answer to the question following the guidelines above."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": regulatory_prompt},
                {"role": "user", "content": user_prompt.format(
                    question=question,
                    content=content
                )},
            ],
            temperature=0.1,
            max_tokens=10000,
        )
        formatted_answer = response.choices[0].message.content.strip()
        logger.info("Formatted answer using LLM regulatory prompt")
        return formatted_answer
    
    def timed(self, query: str, content):
        """Rewriten for content arg"""
        start = time.perf_counter()
        result = self.process_message(query, content)
        duration = time.perf_counter() - start

        return result, duration
