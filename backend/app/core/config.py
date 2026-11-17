import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

class Settings:
    PROJECT_NAME: str = "Fashion RAG — Multimodal AI Assistant"
    VERSION: str = "2.0.0"
    DESCRIPTION: str = "Multimodal Retrieval-Augmented Generation for Fashion E-Commerce"
    API_PREFIX: str = "/api"
    
    # Security & Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-fashion-rag-jwt-token-key-2026-secure")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'fashion_rag.db')}")
    
    # Hardware & Model Settings
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_REPO: str = "minhunhooo/flan-t5-large-fashion-rag"
    CLIP_CKPT: str = "openai/clip-vit-base-patch32"
    BLIP_MODEL_ID: str = "Salesforce/blip-image-captioning-base"
    HF_SPACE_REPO: str = "minhunhooo/fashion-rag-demo"
    DATASET_NAME: str = "ashraq/fashion-product-images-small"
    
    # Local Caching Paths
    DATA_CACHE_DIR: str = CACHE_DIR
    FAISS_PATH: str = (
        os.path.join(BASE_DIR, "fashion_clip_image.faiss")
        if os.path.exists(os.path.join(BASE_DIR, "fashion_clip_image.faiss"))
        else os.path.join(CACHE_DIR, "fashion_clip_image.faiss")
    )
    META_PATH: str = (
        os.path.join(BASE_DIR, "fashion_image_meta.csv")
        if os.path.exists(os.path.join(BASE_DIR, "fashion_image_meta.csv"))
        else os.path.join(CACHE_DIR, "fashion_image_meta.csv")
    )
    CAPTION_PATH: str = (
        os.path.join(BASE_DIR, "caption_cache.json")
        if os.path.exists(os.path.join(BASE_DIR, "caption_cache.json"))
        else os.path.join(CACHE_DIR, "caption_cache.json")
    )
    
    # Search Defaults
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 10
    SIM_THRESHOLD: float = 0.22
    CANDIDATE_POOL: int = 30
    MAX_NEW_TOKENS: int = 80

settings = Settings()
