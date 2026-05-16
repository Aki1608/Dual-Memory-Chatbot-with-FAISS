# Dual-Memory Chatbot with FAISS

A full-stack AI engineering project demonstrating a custom, enterprise-grade cognitive architecture. Rather than relying on out-of-the-box memory wrappers, this project orchestrates a **Hybrid Memory System** by combining a chronological sliding window with a highly optimized vector database for infinite semantic recall.

## The Engineering Challenge & Solution
Standard Large Language Models (LLMs) suffer from severe context window limitations. 
* Standard "Window" memory forgets long-term facts.
* Standard "RAG" (Retrieval-Augmented Generation) disrupts short-term conversational flow.

**The Solution:** OmniContext AI manually orchestrates a dual-memory pipeline. It maintains a **Short-Term Buffer** (the last 5 messages) for immediate conversational fluidity, while simultaneously embedding every user interaction into a local **FAISS Vector Database**. When queried, the engine retrieves the top 3 semantic historical memories and combines them with the short-term buffer into a unified Master Prompt.

## System Architecture

```text
User Input ───► FAISS Vector Store ───► Top 3 Semantic Memories (Long-Term)
                    (Embedding)                     │
                                                    ▼
User Input ────────► Sliding Buffer ───► Last 5 Messages (Short-Term)
                                                    │
                                                    ▼
                                          Master Prompt Compiler
                                                    │
                                                    ▼
                                            Llama 3.1 Inference
```

## Tech Stack
* **Language:** Python 3
* **AI Orchestration:** LangChain Core (Prompts & Runnables)
* **Inference Engine:** Groq API (`llama-3.1-8b-instant`)
* **Embeddings:** Hugging Face (`all-MiniLM-L6-v2`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Frontend UI:** Gradio 6.0 (Blocks Dashboard)

## Project Structure
* `app.py` - The frontend application. Uses Gradio Blocks to build a custom multi-pane dashboard that visualizes the AI's internal state (retrieved vectors and buffer window) in real-time.
* `advanced_memory_engine.py` - The cognitive backend. Bypasses standard LangChain memory classes to manually orchestrate vector embeddings, similarity search, buffer slicing, and the Master Prompt.
* `requirements.txt` - Project dependencies.

## Quick Start Guide

### 1. Environment Setup
Create an isolated Python sandbox to ensure dependency safety.
```bash
python -m venv venv

# Activate on Mac/Linux
source venv/bin/activate  

# Activate on Windows
venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Requires `langchain`, `langchain-groq`, `langchain-community`, `langchain-huggingface`, `faiss-cpu`, `python-dotenv`, and `gradio>=6.0.0`)*

### 3. API Configuration
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 4. Launch the Dashboard
Start the local server:
```bash
python app.py
```

---

## How to Test the Architecture (The "Amnesia Test")
To prove the dual-memory system works, perform the following stress test in the UI:

1. **Establish a Fact:** Tell the bot a highly specific detail (e.g., *"I am planning to buy a Tata Curvv, specifically the Pure S+ variant."*).
2. **Flush the Buffer:** Ask the bot 6 completely unrelated questions (e.g., math problems, jokes, weather). This guarantees the car fact is pushed entirely out of the 5-message sliding window.
3. **The Proof:** Ask the bot, *"Hey, what car was I planning to buy?"*
4. **Observe the UI:** Watch the Right Panel. You will see the AI's short-term buffer is empty of the car fact, but the FAISS Vector Database will instantly retrieve the semantic memory and successfully answer the question.