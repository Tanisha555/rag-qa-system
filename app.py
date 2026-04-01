# app.py
# -----------------------------------------------------------
# CLI interface for the RAG Q&A system
# Run: python app.py
# -----------------------------------------------------------

from dotenv import load_dotenv
load_dotenv()

import os
from rag_pipeline import (
    load_and_split_document,
    create_vector_store,
    build_rag_chain,
    ask_question,
    evaluate_response
)


# ── CONFIG ──────────────────────────────────────────────────
DOCUMENT_PATH = "data/ai_overview.txt"   # the document we'll ask questions about
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")   # set as env variable (never hardcode!)
GROQ_MODEL    = "llama-3.1-8b-instant"   # free Groq model (llama3-8b-8192 was decommissioned)
TOP_K_CHUNKS  = 3                        # how many chunks to retrieve per question
# ────────────────────────────────────────────────────────────


def main():
    print("\n" + "="*60)
    print("  RAG-Based Q&A System")
    print("  Built with: LangChain + FAISS + HuggingFace + Groq")
    print("="*60 + "\n")

    # Validate API key
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not set.")
        print("Set it with: export GROQ_API_KEY='your_key_here'")
        print("Get a free key at: https://console.groq.com")
        return

    # ── Build the pipeline ──────────────────────────────────
    chunks       = load_and_split_document(DOCUMENT_PATH)
    vector_store = create_vector_store(chunks)
    rag_chain    = build_rag_chain(vector_store, GROQ_API_KEY, GROQ_MODEL, TOP_K_CHUNKS)

    print("\n[4/4] System ready! Type your question below.")
    print("      Commands: 'quit' to exit | 'eval' to toggle evaluation mode\n")

    # ── Interactive Q&A loop ────────────────────────────────
    eval_mode = True   # show retrieved chunks and evaluation label prompt

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "eval":
            eval_mode = not eval_mode
            print(f"Evaluation mode: {'ON' if eval_mode else 'OFF'}\n")
            continue

        print("\nThinking...\n")
        answer, source_docs = ask_question(rag_chain, user_input)

        if eval_mode:
            # Shows answer + retrieved chunks so you can manually label quality
            evaluate_response(answer, source_docs)
        else:
            print(f"Answer: {answer}\n")

        # Log this Q&A pair to a file (simulates AI data collection)
        log_qa_pair(user_input, answer, source_docs, eval_mode)


def log_qa_pair(question, answer, source_docs, eval_mode):
    """
    Save each question-answer pair to a log file.
    This simulates the data collection + annotation workflow used in real AI pipelines.
    You can later review this log and add your quality labels (CORRECT/PARTIAL/etc.)
    """
    os.makedirs("logs", exist_ok=True)
    with open("logs/qa_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\nQUESTION: {question}\n")
        f.write(f"ANSWER: {answer}\n")
        f.write(f"CHUNKS_USED: {len(source_docs)}\n")
        f.write(f"EVALUATION_LABEL: [CORRECT / PARTIAL / INCORRECT / NO_ANSWER]\n")
        f.write("-"*60 + "\n")


if __name__ == "__main__":
    main()
