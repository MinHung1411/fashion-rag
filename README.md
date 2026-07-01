# 🛍️ Fashion RAG — Multimodal Q&A Bot

[![Live Demo](https://img.shields.io/badge/Live_Demo-Try_Now!-brightgreen?style=for-the-badge&logo=huggingface)](https://minhunhooo-fashion-rag-demo.hf.space/)
[![Dataset](https://img.shields.io/badge/Dataset-Fashion_Product_Images-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/datasets/ashraq/fashion-product-images-small)

> **Live Demo:** [Fashion RAG Web App](https://minhunhooo-fashion-rag-demo.hf.space/)

## 📖 Overview
Fashion RAG is an advanced Retrieval-Augmented Generation (RAG) system designed specifically for the fashion e-commerce domain. By combining state-of-the-art vision and language models, the system can understand natural language queries, retrieve the most relevant fashion items, and generate precise, context-aware responses.

## 🚀 Key Features
- **Natural Language Search**: Converts complex user queries into accurate top-K product retrievals.
- **Multimodal Understanding**: Processes both textual product metadata and raw product images.
- **Smart Context Extraction**: Intelligently extracts implicit filters (e.g., gender, color, season) from user queries to narrow down search spaces.
- **High-Performance Architecture**: Features parallel asset downloading, lazy-loaded BLIP captioning, and `torch.compile` optimization for rapid inference.
- **Interactive UI**: Built with Gradio for a seamless and responsive user experience.

## 🏗️ System Architecture
- **Retrieval Engine**: CLIP (`openai/clip-vit-base-patch32`) — Generates joint image and text embeddings paired with **FAISS** for ultra-fast vector similarity search.
- **Image Captioning**: BLIP (`Salesforce/blip-image-captioning-base`) — Employed with a lazy-loading caching mechanism to significantly optimize startup time and reduce VRAM usage.
- **Generation Model**: Fine-tuned Flan-T5-large (`minhunhooo/flan-t5-large-fashion-rag`).
- **Knowledge Base**: [Fashion Product Images Small](https://huggingface.co/datasets/ashraq/fashion-product-images-small) containing over 44,000 curated fashion products.

## 📊 Performance Metrics (Evaluation Set)
| Metric | Score |
|---|---|
| ROUGE-1 | 0.4950 |
| ROUGE-L | 0.4372 |
| BERTScore-F1 | 0.9186 |
| Grounding-rate | 1.0000 |
| **Composite Score** | **0.7907** |

## 🛠️ Technical Highlights
- Implemented multithreading for FAISS index and cache fetching.
- Designed smart memory management by dynamically loading BLIP only during cache misses.
- Complete metadata separation from raw dataset downloads, significantly reducing initialization overhead.

---
*Developed by Ho Minh Hung.*
*This project serves as a comprehensive demonstration of applying Large Language Models (LLMs) and Multimodal RAG techniques to solve real-world e-commerce challenges.*
