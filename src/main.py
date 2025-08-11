from rag_nakamo.settings import get_settings
from rag_nakamo.agents.base import BaseAgent
from rag_nakamo.agents.orchestrator import OrchestratorAgent
from rag_nakamo.agents.rag import RAGAgent
from rag_nakamo.agents.response import ResponseAgent
from rag_nakamo.logger_config import setup_logging
from rag_nakamo.security.prompt_guard import PromptGuard
from rag_nakamo.security.schemas import ClassificationResult
import logging, os

logger = logging.getLogger(__name__)
def main():
    settings = get_settings()
    # logger.info(f"Loaded settings: {settings}")

    # sample query
    query = "What are the requirements for medical device software?"

    # 1. Orchestrator 
    orchestrator = OrchestratorAgent(name="Orchestrator", description="Orchestrates the RAG process")
    # timed version
    action_plan, duration = orchestrator.timed(query)
    # logger.info(f"Generated action plan: {action_plan} in {duration:.2f} seconds.")
    logger.info(f"Generated action plan in {duration:.2f} seconds.")

    # 2. RAG Agent
    rag_agent = RAGAgent(name="RAG Agent", description="Retrieval-Augmented Generation Agent")
    retrieval, duration = rag_agent.timed(action_plan)
    logger.info(f"RAG retrieved {len(retrieval)} sources in {duration:.2f} seconds.")
    # 2b rerank
    if settings.enable_rerank:
        logger.info(f"Reranking enabled. Top {settings.rerank_top_k} results after reranking. Model: {settings.rerank_model}")
    for i in retrieval[:1]:
        logger.info(f"First retrieval : {i['relevance_score']:.4f} for {i['source']} at page {i['page']}:")# {i['content'][:200]}[...]") 
        # logger.info(f"rag retrieval : {i}")

    # 3. Response Agent - After RAG processing
    response_agent = ResponseAgent(name="Response Agent", description="Final response formatting and validation age")
    final_response = response_agent.process_message(query, retrieval)
    logger.info(f"Final answer OK")
    # if timed
    # logger.info(f"ResponseAgent formatted answer in {duration:.2f} seconds.")

    # 4. Guarded Response after response answer
    prompt_guard = PromptGuard()
    # 4a guard classification
    guard_response = prompt_guard._call_classifier(
        user_prompt=query,
        model_response=final_response, 
        context_docs=retrieval
    )
    logger.info(f"Guard response: {guard_response}")
    # 4b decision
    classification = ClassificationResult(
            prompt_harm_label=guard_response.get("prompt_harm_label","harmful"),
            response_refusal_label=guard_response.get("response_refusal_label","compliance"),
            response_harm_label=guard_response.get("response_harm_label","harmful"),
        )
    decision = prompt_guard._decide(classification)
    logger.info(f"Guard decision: {decision.status} - {decision.reason}")
    # final
    guarded_response = prompt_guard.classify_and_decide(user_prompt=query, draft_answer=final_response, context_docs=retrieval)
    logger.info(f"Guarded final response: OK")
    # logger.info(f"Guarded final response: {guarded_response.final_answer}")

if __name__ == "__main__":
    setup_logging()
    main()