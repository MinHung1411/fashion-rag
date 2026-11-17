import uvicorn
import os
import sys

def main():
    print("=" * 65)
    print("🛍️  FASHION RAG — MULTIMODAL AI ASSISTANT  (v2.0)")
    print("   Built by: Ho Minh Hung")
    print("   FastAPI Backend + CLIP ViT-B/32 + FAISS + Flan-T5")
    print("=" * 65)
    print("🚀 Server starting...")
    print("🌐 Web Application : http://localhost:8000")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("=" * 65)
    
    uvicorn.run(
        "backend.app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )

if __name__ == "__main__":
    main()
