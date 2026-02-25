import os
from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import chromadb

# optional .env loader
from dotenv import load_dotenv
load_dotenv()

def ingest_docs(doc_folder="docs", persist_dir="chroma_db"):
    loader = DirectoryLoader(doc_folder, glob="**/*")
    docs = loader.load()

    # chunk text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(docs)

    # local embeddings via Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

    client = chromadb.Client(persist_directory=persist_dir)
    collection = client.create_collection(name="local_rag_docs")

    for d in docs:
        vector = embeddings.embed_query(d.page_content)
        collection.add(
            ids=[f"{d.metadata.get('source','doc')}_{hash(d.page_content)}"],
            documents=[d.page_content],
            embeddings=[vector],
            metadatas=[d.metadata]
        )

    print("📌 Ingested", len(docs), "document chunks.")

if __name__ == "__main__":
    ingest_docs()
