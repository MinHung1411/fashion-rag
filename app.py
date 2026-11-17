"""
Fashion RAG — Multimodal AI Assistant (v2.0)
Re-architected with FastAPI, JWT Auth, Conversation History, and Modern UI.
Author: Ho Minh Hung
"""

import os
import uvicorn
import gradio as gr
from backend.app.main import app

# Mount Gradio blocks at /gradio for backward-compatibility with Hugging Face Space Gradio SDK
with gr.Blocks(title="Fashion RAG Multimodal AI") as demo:
    gr.HTML(
        """
        <iframe src="/" style="position:fixed; top:0; left:0; bottom:0; right:0; width:100%; height:100%; border:none; margin:0; padding:0; overflow:hidden; z-index:999999;">
        </iframe>
        """
    )

try:
    app = gr.mount_gradio_app(app, demo, path="/gradio")
except Exception as e:
    print(f"Notice mounting gradio: {e}")

if __name__ == "__main__":
    # Hugging Face Spaces always exposes port 7860 for web apps
    port = 7860
    print("=" * 65)
    print("🛍️  FASHION RAG — MULTIMODAL AI ASSISTANT  (v2.0)")
    print("   Author: Ho Minh Hung")
    print(f"🚀 Server listening on http://0.0.0.0:{port}")
    print("=" * 65)
    uvicorn.run(app, host="0.0.0.0", port=port)
