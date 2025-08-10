import re
from glob import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from rag_nakamo.settings import get_settings
from rag_nakamo.vectorstore.chroma_manager import create_and_populate_vector_store, get_vector_store_retriever

def load_pdfs(data_dir):
    """Load all PDFs from a directory"""
    # print(data_dir)
    pdf_files = glob(data_dir + "*.pdf")
    documents = []
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        pages = load_pdf(pdf_file)
        documents.extend(pages)
        print(f"Added {len(pages)} pages")

    return documents

def load_pdf(file_path):
    """Load a single PDF file"""
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    for i, page in enumerate(pages):
        page.metadata.update({
            "source": file_path,
            "file_path": str(file_path),
            "page_number": i
        })
    return pages

def chunk_documents(documents, chunker):
    """Split documents now using semantic chunking"""
    chunks = []
    for doc in documents:
        doc_chunks = chunker.create_documents([doc.page_content])
        for chunk in doc_chunks:
            chunk.metadata = doc.metadata.copy()
        chunks.extend(doc_chunks)

    return chunks

def analyze_chunks(chunks):
    """Basic chunk analysis"""
    word_counts = [len(chunk.page_content.split()) for chunk in chunks]
    
    print("\nChunk Analysis:")
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg words per chunk: {sum(word_counts)/len(word_counts):.1f}")
    print(f"Min words: {min(word_counts)}")
    print(f"Max words: {max(word_counts)}")
    print(f"Sources: {len(set(chunk.metadata['source'] for chunk in chunks))}")
    
def test_retrieval(embeddings, chroma_db_path):
    """Test document retrieval"""
    retriever = get_vector_store_retriever(embeddings, chroma_db_path)
    
    test_queries = [
        "What are the requirements for medical device software?",
        "How should risk management be documented?",
        "What are the validation requirements?"
    ]
    
    for query in test_queries:
        results = retriever.invoke(query)
        print(f"\nQuery: '{query}'")
        print(f"Retrieved {len(results)} results")
        if results:
            top_result = results[0]
            print(f"Top result from: {top_result.metadata.get('source', 'unknown')}")
            print(f"Preview: {top_result.page_content[:150]}...")

# def main(data_dir="data/"):
#     print("Starting document processing...")
    
#     # Initialize semantic chunker
#     settings = get_settings()
#     embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
#     chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    
#     # Load and process documents
#     documents = load_pdfs(data_dir)
#     print(f"Loaded total of {len(documents)} pages from {data_dir}")
#     chunks = chunk_documents(documents, chunker)
    
#     # Show results
#     analyze_chunks(chunks)
    
#     print("\nFirst chunk preview:")
#     print(chunks[0].page_content[:200] + "...")


def main(data_dir="data/", chroma_db_path="./chroma_db"):
    print("Starting document processing...")
    settings = get_settings()
    embeddings = OpenAIEmbeddings(api_key=settings.openai_api_key)
    # semantic chunker
    chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile") #

    # Load and chunk pdfs
    documents = load_pdfs(data_dir)
    print(f"Loaded total of {len(documents)} pages from {data_dir}")
    chunks = chunk_documents(documents, chunker)
    
    # Show results
    analyze_chunks(chunks)
    
    print("\nFirst chunk preview:")
    print(chunks[0].page_content[:200] + "...")

    print("\n=== 1. Creating and populating vector store ===")
    # vector store
    vector_store = create_and_populate_vector_store(
        chunks=chunks,
        embeddings=embeddings,
        chroma_db_path=chroma_db_path
    )

    # Test retrieval
    print("\n=== Testing retrieval ===")
    test_retrieval(embeddings, chroma_db_path)

if __name__ == "__main__":
    # Use absolute path to ensure we find the files
    # data_path = os.path.join(os.path.dirname(__file__), "data/")
    main()