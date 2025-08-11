from rag_nakamo.settings import get_settings
from rag_nakamo.agents.base import BaseAgent
from rag_nakamo.agents.orchestrator import OrchestratorAgent
from rag_nakamo.agents.rag import RAGAgent
from rag_nakamo.logger_config import setup_logging
from rag_nakamo.security.prompt_guard import PromptGuard
from rag_nakamo.security.schemas import ClassificationResult
import logging, os

logger = logging.getLogger(__name__)
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
    retrieval, duration = rag_agent.timed(action_plan)
    logger.info(f"RAG retrieved {len(retrieval)} sources in {duration:.2f} seconds.")
    if settings.enable_rerank:
        logger.info(f"Reranking enabled. Top {settings.rerank_top_k} results after reranking. Model: {settings.rerank_model}")
    for i in retrieval[:1]:
        # print(i)
        logger.info(f"{i['relevance_score']:.4f} for {i['source']} at page {i['page']}: {i['content'][:200]}[...]")  # Log first 100 chars

    # 3. Prompt Guard
    logger.info("Running prompt guard safety checks...")
    prompt_guard = PromptGuard()
    context_snippet = prompt_guard._build_context_snippet(retrieval)
    logger.info(f"Context snippet for safety check: OK") #
    # safety classification
    guard_response = prompt_guard._call_classifier(
        user_prompt=query,
        model_response=retrieval[0],  # Use first retrieved document as draft answer
        context_docs=retrieval  # pass retrieved docs for context   
    )
    logger.info(f"Guard response: {guard_response}")
    classification = ClassificationResult(
            prompt_harm_label=guard_response.get("prompt_harm_label","harmful"),
            response_refusal_label=guard_response.get("response_refusal_label","compliance"),
            response_harm_label=guard_response.get("response_harm_label","harmful"),
        )
    decision = prompt_guard._decide(classification)
    logger.info(f"Guard decision: {decision.status} - {decision.reason}")

    guarded_response = prompt_guard.classify_and_decide(
        user_prompt=query,
        draft_answer=retrieval[0]['content'],  # Use first retrieved document as draft answer
        context_docs=retrieval  # pass retrieved docs for context
    )
    logger.info(f"Guarded complete response: {guarded_response.final_answer}")

if __name__ == "__main__":
    setup_logging()
    main()