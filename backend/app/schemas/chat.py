from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from .rag import ProductItem, GroundingInfo, LatencyInfo

class ConversationCreate(BaseModel):
    title: Optional[str] = "Cuộc trò chuyện mới"

class ConversationUpdate(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None

class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)
    gender_filter: Optional[str] = None
    category_filter: Optional[str] = None

class MessageOut(BaseModel):
    id: int
    conversation_id: str
    sender: str
    content: str
    image_url: Optional[str] = None
    is_outfit: Optional[bool] = False
    retrieved_products: Optional[List[Dict[str, Any]]] = []
    grounding_status: Optional[str] = "none"
    matched_terms: Optional[List[str]] = []
    filters: Optional[Dict[str, Any]] = {}
    latency: Optional[Dict[str, Any]] = {}
    created_at: datetime

    class Config:
        from_attributes = True

class ConversationOut(BaseModel):
    id: str
    title: str
    pinned: bool
    created_at: datetime
    updated_at: datetime
    message_count: int = 0
    last_message: Optional[str] = None

    class Config:
        from_attributes = True

class ConversationDetail(ConversationOut):
    messages: List[MessageOut] = []
