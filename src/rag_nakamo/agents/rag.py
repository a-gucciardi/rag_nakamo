from rag_nakamo.agents.base import BaseAgent
from rag_nakamo.settings import get_settings
from rag_nakamo.vectorstore.chroma_manager import get_vector_store_retriever
from openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
import logging, json

class RAGAgent(BaseAgent):
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.client_type = "OpenAI"
        self.embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key, model="text-embedding-3-large")
        self.retriever = get_vector_store_retriever(
            embeddings=self.embeddings,
            chroma_db_path=settings.chroma_db_path,
            search_kwargs={"k": settings.retrieval_top_k}
        )

    def process_message(self, query: str, focus_areas: list = None):
        """ Here query is the action plan from orchestrator """
        orch_query = query.get("original_question", "")
        # focus_areas = query.get("arguments", {}).get("focus_areas", []) # currently unused
        retrieved = self.search_documents(orch_query)
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
        # retrieved_docs = self.retriever.invoke(query)
        # returns tuples of (document, score)
        docs_scores = self.retriever.vectorstore.similarity_search_with_score(
            query, 
            k=get_settings().retrieval_top_k
        )
            # Add scores to document metadata
        retrieved_docs = []
        for doc, score in docs_scores:
            doc.metadata["score"] = score
            retrieved_docs.append(doc)
        return retrieved_docs
        # return self._format_search_results(retrieved_docs, query)