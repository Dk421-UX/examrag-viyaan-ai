📚 ExamRAG — Powered by Viyaan AI

AI-powered exam preparation system built using Retrieval-Augmented Generation (RAG).

---

🚀 What it does

ExamRAG helps students turn study material into exam-ready outputs:

- Generate 5-mark and 10-mark answers
- Create revision notes
- Build a searchable knowledge base from PDFs
- Run fully locally (no paid APIs)

---

⚡ Key Features

- Intelligent answer generation using context
- Structured outputs (Keypoints + Exam Tips)
- Multiple modes: Explanation, 5-Mark, 10-Mark, Revision, Quiz, Strategy
- Fast local inference with Ollama
- Modular RAG pipeline (ingest → chunk → embed → retrieve → generate)

---

🧠 Tech Stack

- Python
- Streamlit (UI)
- FAISS (Vector Store)
- Sentence Transformers (Embeddings)
- Ollama (Local LLM)

---

⚙️ How to Run

git clone https://github.com/Dk421-UX/examrag-viyaan-ai
cd examrag-viyaan-ai

pip install -r requirements.txt

ollama pull tinyllama

streamlit run frontend/app.py

---

📌 System Design (High-Level)

User Query
   ↓
Retriever (Vector Search)
   ↓
Relevant Chunks
   ↓
Prompt Builder
   ↓
LLM (Ollama)
   ↓
Structured Answer (Answer + Keypoints + Exam Tip)

---

⚠️ Notes

- Requires Ollama running locally ("ollama serve")
- Designed for CPU systems (~8GB RAM)
- Prototype version focused on learning and experimentation

---

💡 About

Built under Viyaan AI — focused on building practical AI systems for real-world use cases.
