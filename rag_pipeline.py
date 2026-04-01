
# RAG (Retrieval-Augmented Generation) Pipeline
# Uses: HuggingFace Embeddings + FAISS vector store + Groq LLM


import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def load_and_split_document(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Step 1: Load a text document and split it into smaller chunks.

    Why chunking?
    - LLMs have a limited context window (they can't read 100 pages at once)
    - Smaller chunks let us find the MOST RELEVANT piece for any question
    - chunk_overlap ensures we don't lose meaning at chunk boundaries
    """
    print(f"[1/4] Loading document: {file_path}")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    print(f"      Document loaded. Total characters: {len(documents[0].page_content)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,       # max characters per chunk
        chunk_overlap=chunk_overlap, # overlap between consecutive chunks
        separators=["\n\n", "\n", ".", " "]  # try to split on natural boundaries
    )
    chunks = splitter.split_documents(documents)
    print(f"      Split into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return chunks


def create_vector_store(chunks):
    """
    Step 2: Convert each text chunk into a vector (embedding) and store in FAISS.

    What is an embedding?
    - A list of numbers (e.g. 384 numbers) that captures the MEANING of a sentence
    - Similar sentences have similar vectors
    - We use cosine similarity to find the closest match to a user's question

    We use a FREE HuggingFace model — no API key needed for this step!
    """
    print("[2/4] Generating embeddings and building FAISS vector store...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
        # Small (80MB), fast, and surprisingly good for retrieval tasks
    )
    vector_store = FAISS.from_documents(chunks, embedding_model)
    print(f"      Vector store ready. Stored {len(chunks)} chunk vectors.")
    return vector_store


def build_rag_chain(vector_store, groq_api_key: str, model_name: str = "llama3-8b-8192", k: int = 3):
    """
    Step 3: Connect the retriever + LLM into a RetrievalQA chain.

    How RAG works at query time:
    1. User asks a question
    2. Question is converted to an embedding
    3. FAISS finds the top-k most similar chunks (retrieval)
    4. Those chunks are injected into the prompt as "context"
    5. Groq LLM reads context + question and generates an answer

    k = how many chunks to retrieve (3 is a good default)
    """
    print("[3/4] Building RAG chain with Groq LLM...")

    llm = ChatGroq(
        api_key=groq_api_key,
        model_name=model_name,   # Free Groq models: llama3-8b-8192, mixtral-8x7b-32768
        temperature=0.2          # Low temperature = more factual, less creative
    )

    # Custom prompt — this is the "prompt engineering" part!
    # We explicitly tell the model to use ONLY the provided context
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are a helpful AI assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."
Do not make up information.

Context:
{context}

Question: {question}

Answer:"""
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",   # "stuff" = concatenate all retrieved chunks into one prompt
        retriever=retriever,
        return_source_documents=True,   # return which chunks were used (great for evaluation!)
        chain_type_kwargs={"prompt": prompt_template}
    )

    print("      RAG chain ready!")
    return rag_chain


def ask_question(rag_chain, question: str):
    """
    Step 4: Run a question through the RAG pipeline and return the answer + sources.
    """
    result = rag_chain.invoke({"query": question})

    answer = result["result"]
    source_docs = result["source_documents"]

    return answer, source_docs


def evaluate_response(answer: str, source_docs: list):
    """
    BONUS: Simple manual evaluation helper.

    In real Gen AI jobs, evaluators label responses as:
    - CORRECT    : Answer is accurate and grounded in the source
    - PARTIAL    : Answer is partly correct or missing details
    - INCORRECT  : Answer is wrong or hallucinated (not in context)
    - NO_ANSWER  : Model correctly said it doesn't know

    This function shows the retrieved chunks so YOU can verify
    whether the answer is grounded — exactly like data evaluation tasks!
    """
    print("\n" + "="*60)
    print("ANSWER:")
    print(answer)
    print("\n" + "-"*60)
    print(f"RETRIEVED CONTEXT (top {len(source_docs)} chunks used):")
    for i, doc in enumerate(source_docs, 1):
        print(f"\n  [Chunk {i}]")
        print(f"  {doc.page_content[:300]}...")
    print("="*60)
    print("\nEVALUATION: Is this answer grounded in the context above?")
    print("  Label options: CORRECT | PARTIAL | INCORRECT | NO_ANSWER")
