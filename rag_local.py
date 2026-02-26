from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def build_local_rag():

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    # --------------------------------------------------
    # Vector Store
    # --------------------------------------------------
    vectordb = Chroma(
        persist_directory="chroma_db",
        collection_name="local_rag_docs",
        embedding_function=embeddings
    )

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    # --------------------------------------------------
    # LLM
    # --------------------------------------------------
    llm = ChatOllama(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        temperature=0,
    )

    # --------------------------------------------------
    # Prompt
    # --------------------------------------------------
    prompt = ChatPromptTemplate.from_template(
        """You are a technical maintenance assistant.

Use ONLY the provided context from the docs folder.

If the answer is not available in the context, say:
"I cannot find this information in the provided documents."

Be concise and precise.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    # --------------------------------------------------
    # Format Documents
    # --------------------------------------------------
    def format_docs(docs):
        formatted_chunks = []
        sources = []

        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            sources.append(source)

            formatted_chunks.append(
                f"[Source {i+1}: {source}]\n{doc.page_content}"
            )

        return {
            "context": "\n\n".join(formatted_chunks),
            "sources": list(set(sources)),  # deduplicated
            "documents": docs
        }

    # --------------------------------------------------
    # Retrieval Step
    # --------------------------------------------------
    def retrieve_and_format(inputs):
        question = inputs["question"]
        docs = retriever.invoke(question)
        return {
            "question": question,
            **format_docs(docs)
        }

    retrieval_chain = RunnableLambda(retrieve_and_format)

    # --------------------------------------------------
    # Generation Step
    # --------------------------------------------------
    generation_chain = (
        {
            "context": lambda x: x["context"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # --------------------------------------------------
    # Final RAG Chain
    # --------------------------------------------------
    def combine_output(inputs):
        answer = generation_chain.invoke(inputs)
        return {
            "answer": answer,
            "sources": inputs["sources"],
            "documents": inputs["documents"],
        }

    rag_chain = retrieval_chain | RunnableLambda(combine_output)

    return rag_chain
