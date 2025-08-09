#!/usr/bin/env python3
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer
from glob import glob

from vector_store_manager import create_and_populate_vector_store, get_vector_store_retriever


# Ingestion script for regulatory PDF documents.
# -> loads the 3 regulatory PDFs and creates a ChromDB vector store.

class RegulatoryEmbeddings(Embeddings):
    """
    Custom LangChain compatible embedding model optimized for regulatory documents.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        all-MiniLM-L6-v2 is a good general-purpose model that handles technical text well.
        """
        self.model = SentenceTransformer(model_name)

    # rename necessary to avoid conflict with langchain's Embeddings
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

    # doesnt work without enmbed_query
    def embed_query(self, text):
        return self.model.encode(text, convert_to_tensor=False).tolist()

def load_regulatory_pdfs(data_dir):
    """
    Load all regulatory PDFs from the data directory
    """
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    all_chunks = []
    pdf_files = glob(os.path.join(data_dir, "*.pdf"))
    # print(f"Found PDF files: {pdf_files}")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pages = loader.load()
        for page in pages:
            page.metadata.update({
                # "document_type": document_type,
                "source": os.path.basename(pdf_file)
            })
        chunks = text_splitter.split_documents(pages)
        all_chunks.extend(chunks)
        print(f"Loaded {len(pages)} pages from {pdf_file}, created {len(chunks)} chunks")

    print(f"Total documents loaded: {len(all_chunks)} chunks from {len(pdf_files)} PDFs")

    return all_chunks

def ingest_regulatory_documents(data_directory = "./data", chroma_db_path = "./chroma_db"):
    """
    Ingest regulatory PDF documents into the vector store.
    
    Args:
        data_directory: Path to directory containing PDF files
        chroma_db_path: Path to ChromaDB storage
    """
    
    pdf_files = glob(os.path.join(data_directory, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {data_directory}:")
    for pdf_file in pdf_files:
        print(f"->{pdf_file}")

    print("\n=== 1. Loading PDF documents ===")
    docs = load_regulatory_pdfs(data_directory)

    # Initialize embeddings
    print("\n=== 2. Initializing embeddings ===")
    # embeddings = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = RegulatoryEmbeddings()
    print("Embeddings initialized.")

    # Create vector store
    print("\n=== 3. Creating Chroma vector store ===")
    vector_store = create_and_populate_vector_store(
        chunks=docs,
        embeddings=embeddings,
        chroma_db_path=chroma_db_path,
        collection_name="regulatory_documents"
    )

    print("\n=== Ingestion of regulatory_documents completed! ===")
    print(f"in: {chroma_db_path} with total chunks: {len(docs)}")

    # retriever test
    print("\n=== Testing retriever ===")
    retriever = get_vector_store_retriever(embeddings, chroma_db_path)
    test_query = "What are the requirements for medical device software?"
    test_results = retriever.invoke(test_query)
    print(f"Test query: '{test_query}'")
    print(f"Retrieved {len(test_results)} relevant documents.")
    print(f"Top result preview: {test_results[0].page_content[:200]}...")

if __name__ == "__main__":
    ingest_regulatory_documents()

