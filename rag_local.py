from langchain.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

def build_local_rag():
    # local embeddings
    embed_model = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

    vectordb = Chroma(persist_directory="chroma_db",
                      collection_name="local_rag_docs",
                      embedding_function=embed_model)

    # local chat LLM via Ollama
    llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434")

    # RAG chain with history
    rag_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                       retriever=vectordb.as_retriever())
    return rag_chain
