from vector_store_manager import get_vector_store_retriever
from ingest import RegulatoryEmbeddings

class RAGAgent():
    """
    The RAG agent searches through regulatory PDFs for relevant information.
    """
    def __init__(self, chroma_db_path: str = "./chroma_db"):
        self.message_history = []
        self.chroma_db_path = chroma_db_path
        self.retriever = get_vector_store_retriever(
            embeddings=RegulatoryEmbeddings(),
            chroma_db_path=self.chroma_db_path,
            collection_name="regulatory_documents",
            search_kwargs={"k": 5}  # Default to top 5 results
        )

    def process_message(self, message):
        """
        Process a search request after orchestrator and return relevant regulatory information. Requires ingest.
        message : action plan from orchestrator
        """
        self.message_history.append(message)
        # parse action plan from orchestrator
        query = message.get("arguments", {}).get("query", "")
        focus_areas = message.get("arguments", {}).get("focus_areas", [])
        # print(query)
        # print(focus_areas)

        # vector similarity search with focus areas
        retrieved_docs = self.retriever.invoke(query)
        if focus_areas:
            retrieved_docs = self._filter_by_focus_areas(retrieved_docs, focus_areas)

        # formatting the retrieved information
        search_results = self._format_search_results(retrieved_docs, query)
        response_data = {
            "query": query,
            "results": search_results,
            "num_results": len(retrieved_docs),
            "original_question": message.get("original_question", query)
        }

        return response_data

    def _filter_by_focus_areas(self, docs, focus_areas):
        # filter documents based on focus areas
        filtered_docs = []
        focus_areas_lower = [area.lower() for area in focus_areas]

        for doc in docs:
            doc_content = doc.page_content.lower()
            doc_source = doc.metadata.get("source", "").lower()

            # if focus area appears in the document
            if any(area in doc_content or area in doc_source for area in focus_areas_lower):
                filtered_docs.append(doc)

        # if no documents with focus areas -> return original docs
        return filtered_docs if filtered_docs else docs

    def _format_search_results(self, docs, query=None):
        # retrieved documents into structured search results
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "")
            doctype = "fda" if "fda" in source.lower() else "who" if "who" in source.lower() else "other"
            # if doc.metadata.get("score") is not None:
            #     relevance_score = doc.metadata.get("score")
            # else:
            #     relevance_score = 0.0
            result = {
                "rank": i + 1,
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                # "relevance_score": relevance_score,
                "document_type": doctype,
            }
            results.append(result)

        return results

    def search_documents(self, query: str, top_k: int = 5):
        # return top_k results
        self.retriever.search_kwargs = {"k": top_k}
        retrieved_docs = self.retriever.invoke(query)

        return self._format_search_results(retrieved_docs, query)
