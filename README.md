# RAG-Based Q&A System 🔍

A beginner-friendly **Retrieval-Augmented Generation (RAG)** pipeline built with LangChain, FAISS, HuggingFace Embeddings, and Groq LLM.

Ask questions from any text document — the system finds the most relevant chunks and uses an LLM to generate grounded answers.

---

## 🧠 What is RAG?

Traditional LLMs answer from memory (their training data). RAG fixes this by:
1. **Retrieving** relevant text chunks from your own documents
2. **Injecting** those chunks into the prompt as context
3. **Generating** an answer grounded only in that context

This reduces hallucinations and lets you use private or up-to-date documents.

---

## 📁 Project Structure

```
rag-qa-system/
│
├── rag_pipeline.py      # Core RAG logic (load → embed → retrieve → answer)
├── app.py               # CLI interface to chat with the system
├── requirements.txt     # All dependencies
│
├── data/
│   └── ai_overview.txt  # Sample document (about AI/Gen AI concepts)
│
└── logs/
    └── qa_log.txt       # Auto-generated Q&A log for evaluation
```

---

## ⚙️ How It Works (Step by Step)

```
Your Document (ai_overview.txt)
        │
        ▼
[1] Text Splitting      → Break document into 500-char chunks
        │
        ▼
[2] Embeddings          → Convert each chunk to a vector (HuggingFace)
        │
        ▼
[3] FAISS Vector Store  → Store all vectors for fast similarity search
        │
        ▼
   User asks a question
        │
        ▼
[4] Retrieve Top-K      → Find 3 most similar chunks using cosine similarity
        │
        ▼
[5] Prompt + LLM        → Inject chunks into prompt → Groq generates answer
        │
        ▼
[6] Log + Evaluate      → Save Q&A pair for manual quality labeling
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/rag-qa-system.git
cd rag-qa-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
- Go to [console.groq.com](https://console.groq.com)
- Sign up and generate a free API key

### 4. Set your API key
```bash
# Linux / Mac
export GROQ_API_KEY="your_key_here"

# Windows (Command Prompt)
set GROQ_API_KEY=your_key_here
```

### 5. Run the app
```bash
python app.py
```

### 6. Ask questions!
```
You: What is RAG?
You: How does prompt engineering work?
You: What is FAISS used for?
You: quit
```

---

## 🔍 Evaluation Mode

The system has a built-in **evaluation mode** (on by default) that shows:
- The generated answer
- The exact document chunks that were retrieved
- A label prompt: `CORRECT | PARTIAL | INCORRECT | NO_ANSWER`

This simulates the **AI data evaluation** workflow used in real Gen AI teams (including RLHF pipelines).

All Q&A pairs are logged to `logs/qa_log.txt` so you can review and annotate them later.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| LLM (text generation) | Groq (LLaMA 3 8B) — free |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` — free, local |
| Vector Store | FAISS — local, no cloud needed |
| Orchestration | LangChain |
| Language | Python 3.9+ |

---

## 💡 Key Concepts Demonstrated

- **Chunking** — splitting documents to fit LLM context windows
- **Vector embeddings** — turning text into numbers that capture meaning
- **Cosine similarity search** — finding the most relevant chunks
- **Prompt engineering** — designing prompts that keep the model grounded
- **Data annotation** — manually labeling AI outputs for quality evaluation
- **RAG pipeline** — connecting retrieval + generation end-to-end

---

## 📝 Sample Questions to Try

The included document (`data/ai_overview.txt`) covers AI/Gen AI concepts. Try:

- *"What is Retrieval-Augmented Generation?"*
- *"How does FAISS work?"*
- *"What is data annotation in AI?"*
- *"What are the types of prompting techniques?"*
- *"What is hallucination in AI?"*
- *"How is RLHF used in Generative AI?"*

---

## 🔧 Customization

To use your own document, replace `data/ai_overview.txt` with any `.txt` file and update `DOCUMENT_PATH` in `app.py`.

---

*Built as part of a Gen AI learning project. Technologies: LangChain · FAISS · HuggingFace · Groq*
