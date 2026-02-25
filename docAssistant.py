import streamlit as st
from rag_local import build_local_rag

st.title("💬 Local RAG Chatbot (Local LLM + Chroma)")

if "history" not in st.session_state:
    st.session_state.history = []

local_rag = build_local_rag()

query = st.text_input("Ask about your manuals or domain docs:")

if st.button("Ask"):
    if query:
        result = local_rag({"question": query, "chat_history": st.session_state.history})
        answer = result["answer"]

        st.session_state.history.append((query, answer))

        st.write("**Answer:**", answer)

        if "source_documents" in result:
            st.write("📑 *Relevant Context:*")
            for doc in result["source_documents"]:
                st.write("-", doc.page_content[:200], "...")

if st.session_state.history:
    st.write("### Conversation History")
    for i, (q, a) in enumerate(st.session_state.history):
        st.write(f"**Q{i+1}:** {q}")
        st.write(f"**A{i+1}:** {a}")
