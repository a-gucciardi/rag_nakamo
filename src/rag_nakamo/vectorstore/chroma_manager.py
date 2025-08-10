import chromadb
from langchain_chroma import Chroma

# Old vector store manager code

def create_and_populate_vector_store(
    chunks, 
    embeddings,
    chroma_db_path="./chroma_db",
    collection_name="regulatory_documents"
):
    persistent_client = chromadb.PersistentClient(path=chroma_db_path)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=persistent_client,
        collection_name=collection_name
    )

    print(f"Vector store created and populated with {len(chunks)} document chunks.")
    return vector_store

def get_vector_store_retriever(
    embeddings,
    chroma_db_path="./chroma_db",
    collection_name="regulatory_documents",
    search_kwargs=None
):
    if search_kwargs is None:
        search_kwargs = {"k": 5}

    persistent_client = chromadb.PersistentClient(path=chroma_db_path)

    vector_store = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    return vector_store.as_retriever(search_kwargs=search_kwargs)