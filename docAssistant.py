import streamlit as st
import logging
from rag_local import build_local_rag

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="💬 Maintenance Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("💬 Maintenance Assistant")

# --------------------------------------------------
# Logging (production safe)
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# Load RAG (cached once per session)
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading knowledge base...")
def load_rag():
    return build_local_rag()

local_rag = load_rag()

# --------------------------------------------------
# Session State
# --------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = []

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.sources = []
        st.rerun()

    st.divider()
    st.caption("RAG Status: ✅ Ready")

# --------------------------------------------------
# Render Chat History
# --------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --------------------------------------------------
# User Input
# --------------------------------------------------
query = st.chat_input("Ask about your manuals or domain docs...")

if query:
    # Add user message immediately
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    try:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            # Call structured RAG
            result = local_rag.invoke({"question": query})

            answer = result["answer"]
            sources = result.get("sources", [])

            response_placeholder.markdown(answer)

        # Save assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        # Save sources separately
        st.session_state.sources = sources

        # Rerun for clean state rendering
        st.rerun()

    except Exception as e:
        logging.exception("RAG error")
        st.error("⚠️ Something went wrong while generating the response.")

# --------------------------------------------------
# Sources Panel
# --------------------------------------------------
if st.session_state.sources:
    with st.expander("📚 Sources & Retrieved Documents", expanded=False):
        for i, source in enumerate(st.session_state.sources, 1):
            st.markdown(f"**Source {i}:** {source}")
