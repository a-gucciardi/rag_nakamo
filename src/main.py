from rag_nakamo.settings import get_settings
from rag_nakamo.agents.base import BaseAgent
from rag_nakamo.agents.orchestrator import OrchestratorAgent
from rag_nakamo.agents.rag import RAGAgent
from rag_nakamo.logger_config import setup_logging
import logging

logger = logging.getLogger(__name__)
def main():
    settings = get_settings()
    # logger.info(f"Loaded settings: {settings}")

    # 1. Orchestrator 
    orchestrator = OrchestratorAgent()

    # sample query
    query = "What are the requirements for medical device software?"
    # logger.info(f"Processing query: {query}") # duplicated inside agent
    # action_plan = orchestrator.process_message(query)
    # logger.info(f"Generated action plan: {action_plan}")

    # timed version
    action_plan, duration = orchestrator.timed(query)
    logger.info(f"Generated action plan: {action_plan}")
    logger.info(f"Processed query in {duration:.2f} seconds.")
    
    # 2. RAG Agent
    rag_agent = RAGAgent(name="RAG Agent", description="Retrieval-Augmented Generation Agent")
    retrieval = rag_agent.process_message(action_plan)
    logger.info(f"Retrieved information: {retrieval[0]}")
    logger.info(f"Retrieved information: {retrieval[1]['relevance_score']:.4f} for {retrieval[1]['source']} at page {retrieval[1]['page']}")
    logger.info(f"Retrieved information: {retrieval[2]['relevance_score']:.4f} for {retrieval[2]['source']} at page {retrieval[2]['page']}") # Distance = 0: Perfect match

if __name__ == "__main__":
    setup_logging()
    main()