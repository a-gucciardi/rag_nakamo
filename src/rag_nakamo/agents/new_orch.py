"""
Simplified Orchestrator that executes a standard RAG workflow.
"""
import logging
from typing import Dict, Any
from openai import OpenAI
from rag_nakamo.agents.base import BaseAgent
from rag_nakamo.settings import get_settings

logger = logging.getLogger(__name__)

class SimpleOrchestrator(BaseAgent):
    """Simplified orchestrator that executes a standard RAG workflow."""
    
    def __init__(self, name: str = "SimpleOrchestrator", 
                 description: str = "Executes standard RAG workflow"):
        super().__init__(name, description)
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.agents = {}
        self.system_prompt = """
        You are an orchestrator for a medtech regulatory assistant system. 
        Your job is to analyze regulatory questions and decide which agents to use.
        For each query, decide whether to:
        1. Use RAG to search regulatory documents
        2. Generate a response with citations
        """

    def register_agent(self, name: str, agent):
        """Register an agent with the orchestrator. Used for logging."""
        self.agents[name] = agent
        logger.info(f"Registered agent: {name}")

    def process_message(self, query: str) -> Dict[str, Any]:
        """Execute the standard RAG workflow."""
        logger.info(f"Orchestrating query: {query}")
        # Step 1: Decide if we should use RAG (simple check)
        should_use_rag = self._should_use_rag(query)

        if should_use_rag:
            # Execute RAG search
            rag_results = self._execute_rag_search(query)
            logger.info(f"RAG search found {len(rag_results)} sources")
        else:
            # Skip RAG for non-regulatory queries
            rag_results = []
            logger.info("Skipped RAG search - not a regulatory query")

        # Step 2: Always generate response (with or without RAG results)
        response = self._generate_response(query, rag_results)

        return {
            "status": "success",
            "response": response,
            "rag_results": rag_results,
            "used_rag": should_use_rag,
            "sources": [r.get('source') for r in rag_results] if rag_results else []
        }

    def _should_use_rag(self, query: str) -> bool:
        """Mock decision: use RAG for regulatory queries."""
        regulatory_keywords = ['fda', 'who', 'regulation', 'regulation', 'medical device', 'software', 
                              'validation', 'design control', 'requirement', 'guidance', 'standard']
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in regulatory_keywords)

    def _execute_rag_search(self, query: str) -> list:
        """Execute RAG search with the RAG agent."""
        rag_agent = self.agents.get("rag_agent")
        logger.info(f"Executing RAG search for: {query}")

        # Format query for RAG agent
        rag_query = {
            "original_question": query,
            "arguments": {"focus_areas": []}
        }

        results, duration = rag_agent.timed(rag_query)
        logger.info(f"RAG completed in {duration:.2f}s")

        return results if results else []

    def _generate_response(self, query: str, rag_results: list) -> str:
        """Generate response with the Response agent."""
        response_agent = self.agents.get("response_agent")
        if not response_agent:
            # Fallback if no response agent
            if rag_results:
                return f"Found {len(rag_results)} relevant sources for: {query}"
            else:
                return f"No regulatory information found for: {query}"

        logger.info(f"Generating response for: {query}")

        # response agent
        response, duration = response_agent.timed(query, rag_results)
        logger.info(f"Response generated in {duration:.2f}s")

        return response if response else f"Unable to generate response for: {query}"