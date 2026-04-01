
# Streamlit Chat UI for the RAG Q&A System
# Run: streamlit run streamlit_app.py


import os
import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import (
    load_and_split_document,
    create_vector_store,
    build_rag_chain,
    ask_question,
)

load_dotenv()

#  PAGE CONFIG 
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="🔍",
    layout="centered"
)

#  STYLES 
st.markdown("""
<style>
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }
    .source-box {
        background: #f0f4ff;
        border-left: 4px solid #1B4F9C;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.82em;
        color: #333;
        margin-top: 6px;
    }
    .eval-bar {
        display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap;
    }
    .eval-label {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78em;
        font-weight: 600;
        cursor: default;
    }
    .correct   { background:#d4edda; color:#155724; }
    .partial   { background:#fff3cd; color:#856404; }
    .incorrect { background:#f8d7da; color:#721c24; }
    .no-answer { background:#e2e3e5; color:#383d41; }
</style>
""", unsafe_allow_html=True)

#  HEADER 
st.title(" RAG-Based Q&A System")
st.caption("Built with LangChain · FAISS · HuggingFace · Groq")
st.divider()

#  SIDEBAR 
with st.sidebar:
    st.header("⚙️ Settings")

    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Get a free key at console.groq.com"
    )

    model_choice = st.selectbox(
        "Groq Model",
        ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        help="All are free on Groq"
    )

    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=1, max_value=6, value=3,
        help="How many document chunks to use as context"
    )

    show_sources = st.toggle("Show retrieved chunks", value=True)
    show_eval    = st.toggle("Show evaluation labels", value=True)

    st.divider()
    st.header(" Document")
    uploaded_file = st.file_uploader("Upload your own .txt file", type=["txt"])

    if st.button(" Clear chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**How RAG works:**")
    st.markdown("""
1.  Document → split into chunks
2.  Chunks → vector embeddings
3.  Stored in FAISS vector DB
4.  Your question → embedding
5.  Find top-k similar chunks
6.  LLM answers using chunks
""")

#  LOAD PIPELINE 
@st.cache_resource(show_spinner="Building RAG pipeline... (only once)")
def get_rag_chain(api_key, model, k, file_content=None):
    if file_content:
        # Save uploaded file temporarily
        os.makedirs("data", exist_ok=True)
        with open("data/uploaded.txt", "w", encoding="utf-8") as f:
            f.write(file_content)
        doc_path = "data/uploaded.txt"
    else:
        doc_path = "data/ai_overview.txt"

    chunks       = load_and_split_document(doc_path)
    vector_store = create_vector_store(chunks)
    chain        = build_rag_chain(vector_store, api_key, model, k)
    return chain

#  INIT CHAT HISTORY 
if "messages" not in st.session_state:
    st.session_state.messages = []

#  VALIDATION 
if not groq_api_key:
    st.warning("👈 Enter your Groq API key in the sidebar to get started. Get one free at [console.groq.com](https://console.groq.com)")
    st.stop()

#  BUILD PIPELINE 
file_content = None
if uploaded_file:
    file_content = uploaded_file.read().decode("utf-8")
    st.info(f"📄 Using uploaded file: **{uploaded_file.name}**")
else:
    st.info("📄 Using default document: **ai_overview.txt** (about AI & Gen AI concepts)")

try:
    rag_chain = get_rag_chain(groq_api_key, model_choice, top_k, file_content)
except Exception as e:
    st.error(f"Failed to build pipeline: {e}")
    st.stop()

#  DISPLAY CHAT HISTORY 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and show_sources and "sources" in msg:
            with st.expander(f"📚 Retrieved chunks ({len(msg['sources'])} used)"):
                for i, chunk in enumerate(msg["sources"], 1):
                    st.markdown(f"""<div class="source-box">
                        <strong>Chunk {i}</strong><br>{chunk[:400]}...
                    </div>""", unsafe_allow_html=True)

        if msg["role"] == "assistant" and show_eval:
            st.markdown("""<div class="eval-bar">
                <strong style="font-size:0.82em;align-self:center">Evaluate:</strong>
                <span class="eval-label correct"> Correct</span>
                <span class="eval-label partial"> Partial</span>
                <span class="eval-label incorrect"> Incorrect</span>
                <span class="eval-label no-answer"> No Answer</span>
            </div>""", unsafe_allow_html=True)

#  CHAT INPUT 
if prompt := st.chat_input("Ask a question about the document..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, source_docs = ask_question(rag_chain, prompt)
                sources = [doc.page_content for doc in source_docs]
            except Exception as e:
                answer  = f"Error: {e}"
                sources = []

        st.markdown(answer)

        if show_sources and sources:
            with st.expander(f"📚 Retrieved chunks ({len(sources)} used)"):
                for i, chunk in enumerate(sources, 1):
                    st.markdown(f"""<div class="source-box">
                        <strong>Chunk {i}</strong><br>{chunk[:400]}...
                    </div>""", unsafe_allow_html=True)

        if show_eval:
            st.markdown("""<div class="eval-bar">
                <strong style="font-size:0.82em;align-self:center">Evaluate:</strong>
                <span class="eval-label correct"> Correct</span>
                <span class="eval-label partial"> Partial</span>
                <span class="eval-label incorrect">Incorrect</span>
                <span class="eval-label no-answer"> No Answer</span>
            </div>""", unsafe_allow_html=True)

        # Log to file
        os.makedirs("logs", exist_ok=True)
        with open("logs/qa_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\nQUESTION: {prompt}\n")
            f.write(f"ANSWER: {answer}\n")
            f.write(f"CHUNKS_USED: {len(sources)}\n")
            f.write(f"EVALUATION_LABEL: [CORRECT / PARTIAL / INCORRECT / NO_ANSWER]\n")
            f.write("-"*60 + "\n")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
