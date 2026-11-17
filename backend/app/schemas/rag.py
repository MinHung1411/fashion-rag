from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ProductItem(BaseModel):
    id: int
    row_idx: int
    productDisplayName: str
    gender: str = ""
    masterCategory: str = ""
    subCategory: str = ""
    articleType: str = ""
    baseColour: str = ""
    season: str = ""
    usage: str = ""
    score: float = 0.0
    caption: str = ""
    image_url: str = ""

class GroundingInfo(BaseModel):
    grounded: bool
    status: str
    matched_terms: List[str] = []
    reason: str = "OK"

class LatencyInfo(BaseModel):
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0

class AskQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    conversation_id: Optional[str] = None
    gender_filter: Optional[str] = None
    category_filter: Optional[str] = None

class AskQueryResponse(BaseModel):
    answer: str
    query: str
    products: List[ProductItem] = []
    grounding: GroundingInfo
    filters: Dict[str, Any] = {}
    latency: LatencyInfo
    conversation_id: Optional[str] = None

class SystemStatsResponse(BaseModel):
    device: str
    total_vectors: int
    total_products: int
    caption_cache_size: int
    clip_loaded: bool
    t5_loaded: bool
    faiss_loaded: bool
    dataset_loaded: bool
    status: str
