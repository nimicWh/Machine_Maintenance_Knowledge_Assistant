# Builds local vector database from documents.
import os
import uuid
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

def ingest_docs(doc_folder="docs", persist_dir="chroma_db"):
    loader = DirectoryLoader(doc_folder, glob="**/*")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    client = chromadb.Client(
        Settings(
            persist_directory=persist_dir,
            is_persistent=True
        )
    )

    collection = client.get_or_create_collection(name="local_rag_docs")

    for d in docs:
        vector = embeddings.embed_query(d.page_content)
        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[d.page_content],
            embeddings=[vector],
            metadatas=[d.metadata]
        )

    print("📌 Ingested", len(docs), "document chunks.")

if __name__ == "__main__":
    ingest_docs()
