import gradio as gr
from memory_engine import EnterpriseMemoryEngine

engine = EnterpriseMemoryEngine()

def process_chat(user_message, chat_history):
    # Get the AI response and the internal memory states
    ai_response, short_term_ui, long_term_ui = engine.chat(user_message)
    
    # Append to visual chat using Gradio 6.0 dictionary format
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ai_response})
    
    return "", chat_history, short_term_ui, long_term_ui

def reset_chat():
    engine.clear_memory()
    return [], "", "Memory cleared."

# --- UI Dashboard ---
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 Enterprise AI: RAG + Windowing Memory")
    gr.Markdown("This architecture combines a **Sliding Window** (for immediate conversational flow) with a **FAISS Vector Database** (for infinite, searchable long-term recall).")
    
    with gr.Row():
        # Left Panel: Chat
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=550, label="Conversation")
            with gr.Row():
                msg = gr.Textbox(placeholder="Type here...", show_label=False, scale=4)
                clear_btn = gr.Button("Clear Short-Term", scale=1)
                
        # Right Panel: Memory Monitors
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Internal Context Sent to AI")
            long_term_box = gr.Textbox(label="1. Retrieved Vector Data (Long-Term)", lines=8, interactive=False)
            short_term_box = gr.Textbox(label="2. Buffer Window (Short-Term)", lines=8, interactive=False)

    msg.submit(
        fn=process_chat, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot, short_term_box, long_term_box]
    )
    
    clear_btn.click(
        fn=reset_chat, 
        inputs=None, 
        outputs=[chatbot, short_term_box, long_term_box]
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())