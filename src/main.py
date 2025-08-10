from rag_nakamo.settings import get_settings
from rag_nakamo.agents.orchestrator import OrchestratorAgent
from rag_nakamo.agents.base import BaseAgent
from rag_nakamo.logger_config import setup_logging
import logging

logger = logging.getLogger(__name__)
def main():
    settings = get_settings()
    # logger.info(f"Loaded settings: {settings}")

    # 1. Orchestrator 
    orchestrator = OrchestratorAgent()

    # sample query
    query = "Is our AI-powered MRI analysis tool considered a medical device software?"
    # logger.info(f"Processing query: {query}") # duplicated inside agent
    # action_plan = orchestrator.process_message(query)
    # logger.info(f"Generated action plan: {action_plan}")

    # timed version
    action_plan, duration = orchestrator.timed(query)
    logger.info(f"Generated action plan: {action_plan}")
    logger.info(f"Processed query in {duration:.2f} seconds.")

if __name__ == "__main__":
    setup_logging()
    main()