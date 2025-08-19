from rag_nakamo.settings import get_settings
from rag_nakamo.agents.new_orch import SimpleOrchestrator
from rag_nakamo.agents.rag import RAGAgent
from rag_nakamo.agents.response import ResponseAgent
from rag_nakamo.security.prompt_guard import PromptGuard
from rag_nakamo.security.schemas import ClassificationResult
from rag_nakamo.logger_config import setup_logging
import logging


# Test the new orchestrator with basic decision making.

logger = logging.getLogger(__name__)

def main():
    settings = get_settings()
    logger.info("Testing Simple Orchestrator with decision making")

    # Create orchestrator
    orchestrator = SimpleOrchestrator()

    # Register agents
    rag_agent = RAGAgent(name="RAG Agent", description="RAG Agent")
    response_agent = ResponseAgent(name="Response Agent", description="Response Agent")

    orchestrator.register_agent("rag_agent", rag_agent)
    orchestrator.register_agent("response_agent", response_agent)

    # Test queries
    test_queries = [
        "What are FDA software validation requirements?",  # Should use RAG
        "What is the weather today?",  # Should NOT use RAG
        "How do medical device regulations work?",  # Should use RAG
    ]

    for query in test_queries:
        logger.info(f"\n{'='*50}")
        logger.info(f"Testing: {query}")
        logger.info('='*50)

        result = orchestrator.process_message(query)

        if result["status"] == "success":
            logger.info(f"✅ SUCCESS")
            logger.info(f"Used RAG: {result['used_rag']}")
            logger.info(f"Sources found: {len(result['rag_results'])}")
            logger.info(f"Response length: {len(result['response'])} chars")

            # Show decision logic working
            if result['used_rag']:
                logger.info("*Decision*: Regulatory query → Used RAG search")
            else:
                logger.info("*Decision*: Non-regulatory query → Skipped RAG")

            # prompt guard tfor final response
            if result['used_rag']:  # if guard regulatory responses
                logger.info("Applying security guard...")
                prompt_guard = PromptGuard()
                guarded_response = prompt_guard.classify_and_decide(
                    user_prompt=query,
                    draft_answer=result['response'],
                    context_docs=result['rag_results']
                )
                logger.info(f"🛡️  Guard decision: {guarded_response.decision.status} - {guarded_response.decision.reason}")
            else:
                logger.info("Skipped security guard for non-regulatory query")

        else:
            logger.error(f"❌ FAILED: {result.get('error', 'Unknown')}")

    logger.info(f"\n Simple orchestration test complete!")

if __name__ == "__main__":
    setup_logging()
    main()
