from rag_nakamo.agents.base import BaseAgent
from rag_nakamo.settings import get_settings
from rag_nakamo.vectorstore.chroma_manager import get_vector_store_retriever
from openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from sentence_transformers import CrossEncoder
import logging, json, time

logger = logging.getLogger(__name__)

class RAGAgent(BaseAgent):
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.client_type = "OpenAI"
        self.embeddings = OpenAIEmbeddings(api_key=self.settings.openai_api_key, model=self.settings.embeddings_model)
        self.retriever = get_vector_store_retriever(
            embeddings=self.embeddings,
            chroma_db_path=self.settings.chroma_db_path,
            search_kwargs={"k": self.settings.retrieval_top_k}
        )
        self.reranker = CrossEncoder(self.settings.rerank_model)
        logger.info(f"RAG initialized with embeddings: {self.settings.embeddings_model}, reranker: {self.settings.enable_rerank} {self.settings.rerank_model}, retrieval_top_k: {self.settings.retrieval_top_k}")

    def process_message(self, query: str, focus_areas: list = None):
        """ Here query is the action plan from orchestrator """
        orch_query = query.get("original_question", "")
        # focus_areas = query.get("arguments", {}).get("focus_areas", []) # currently unused
        retrieved = self.search_documents(orch_query)
        # rerank after first retrieval
        if self.settings.enable_rerank: retrieved = self.rerank_documents(orch_query, retrieved)
        results = []
        for i, doc in enumerate(retrieved):
            source = doc.metadata.get("source", "")
            result = {
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "relevance_score": doc.metadata.get("score")
            }
            results.append(result)

        return results

    def search_documents(self, query: str):
        # retrieved_docs = self.retriever.invoke(query)-> no score
        # now returns tuples of (document, score)
        docs_scores = self.retriever.vectorstore.similarity_search_with_score(
            query, 
            k=self.settings.retrieval_top_k
        )
        # Add scores to document metadata
        retrieved_docs = []
        for doc, score in docs_scores:
            doc.metadata["score"] = score
            retrieved_docs.append(doc)
        return retrieved_docs

    def rerank_documents(self, query: str, documents: list):
        """ Optional rerank retrieved documents based on relevance to query """
        # Prepare inputs for reranker
        rerank_pairs = [(query, doc.page_content) for doc in documents]
        re_scores = self.reranker.predict(rerank_pairs)

        # document - rescore pairs
        scored_docs = [(i, doc, score) for i, (doc, score) in enumerate(zip(documents, re_scores))] # 
        sorted_docs = sorted(scored_docs, key=lambda x: x[2], reverse=True) # desc order scores, 2: score
        top_docs = [doc for _, doc, _ in sorted_docs[:self.settings.rerank_top_k]] # top k

        # log changes in order
        order_changes = [
            {"change": f"{idx} -> {new_idx} ({float(score):.4f})"}
            for new_idx, (idx, doc, score) in enumerate(sorted_docs)
        ]
        logging.info(f"Reranked documents: {json.dumps(order_changes, indent=2)}")

        return top_docs  # top k