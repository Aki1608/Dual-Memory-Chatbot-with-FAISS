import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class EnterpriseMemoryEngine:
    def __init__(self):
        print("Initializing Enterprise Hybrid Memory Engine...")
        
        # Initialize the LLM (The Brain)
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7,
        )
        
        # Initialize the Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize FAISS Vector Database (Long-Term Memory)
        # We initialize it with a dummy text so FAISS knows the vector dimensions
        self.vector_store = FAISS.from_texts(["[SYSTEM MEMORY INITIALIZED]"], self.embeddings)
        
        # Initialize Short-Term Buffer Window (Short-Term Memory)
        self.short_term_buffer = [] 
        self.window_size = 5 # Keeps the last 5 interactions (10 messages total)

    def chat(self, user_input):
        """Orchestrates RAG retrieval, Window buffering, and AI inference."""
        
        # RETRIEVE LONG-TERM MEMORY
        # Search the vector database for the top 3 past interactions most relevant to the current prompt
        docs = self.vector_store.similarity_search(user_input, k=3)
        long_term_context = "\n".join([doc.page_content for doc in docs])
        
        # GATHER SHORT-TERM MEMORY
        # Format the last 5 messages to maintain immediate conversational flow
        short_term_context = ""
        for msg in self.short_term_buffer:
            short_term_context += f"{msg['role'].capitalize()}: {msg['content']}\n"
            
        # BUILD THE MASTER PROMPT
        system_template = """You are an advanced AI assistant with a dual-memory cognitive architecture.
        
        [LONG-TERM MEMORY (Highly relevant past interactions)]
        {long_term_context}
        
        [SHORT-TERM MEMORY (The immediate past 5 messages)]
        {short_term_context}
        
        Use the context above to inform your response. Do not explicitly say "I am reading from memory" unless asked how you know something.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{input}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({
            "long_term_context": long_term_context,
            "short_term_context": short_term_context,
            "input": user_input
        })
        ai_text = response.content
        
        #UPDATE BOTH MEMORIES FOR NEXT TIME
        # Update Short-Term Buffer
        self.short_term_buffer.append({"role": "user", "content": user_input})
        self.short_term_buffer.append({"role": "ai", "content": ai_text})
        
        # Enforce the sliding window (Slice the list to keep only the newest N messages)
        if len(self.short_term_buffer) > (self.window_size * 2):
            self.short_term_buffer = self.short_term_buffer[-(self.window_size * 2):]
            
        # Update Long-Term Vector Store
        # Save the exact interaction as a searchable document
        interaction_log = f"User asked: {user_input} | AI replied: {ai_text}"
        self.vector_store.add_texts([interaction_log])
        
        # Return data to update the UI
        return ai_text, short_term_context, long_term_context

    def clear_memory(self):
        """Wipes the short-term buffer (Vector store remains persistent for the session)"""
        self.short_term_buffer = []
        return "Short-term buffer wiped. Long-term semantic memory remains intact."