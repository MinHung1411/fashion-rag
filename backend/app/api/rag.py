from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from ..core.database import get_db
from ..schemas.rag import AskQueryRequest, AskQueryResponse, ProductItem, GroundingInfo, LatencyInfo
from ..services.rag_engine import rag_engine
from ..services.catalog_service import catalog_service
from ..services.auth_service import get_optional_user
from ..models.user import User

router = APIRouter(prefix="/rag", tags=["RAG Engine"])

CURATED_EXAMPLES = [
    {
        "query": "Phối cho tôi một set đồ nam công sở lịch lãm",
        "title": "Set đồ nam công sở",
        "category": "AI Stylist • Mix & Match",
        "description": "Tự động kết hợp áo sơ mi, quần âu, giày da và phụ kiện hoàn chỉnh"
    },
    {
        "query": "Do you have casual dresses for women in red?",
        "title": "Váy đầm casual nữ",
        "category": "Nữ • Trang phục",
        "description": "Gợi ý váy đầm dạo phố màu đỏ trẻ trung và hiện đại"
    },
    {
        "query": "What black formal shoes are available for men?",
        "title": "Giày da nam công sở",
        "category": "Nam • Giày dép",
        "description": "Tìm kiếm các mẫu giày tây da màu đen thanh lịch cho nam giới"
    },
    {
        "query": "Show me white sneakers for men",
        "title": "Sneakers trắng năng động",
        "category": "Nam • Sneaker",
        "description": "Tìm kiếm giày thể thao sneaker màu trắng phong cách tối giản"
    }
]

@router.post("/ask", response_model=AskQueryResponse)
def ask_rag(req: AskQueryRequest):
    result = rag_engine.ask(
        query=req.query,
        top_k=req.top_k,
        gender_filter=req.gender_filter,
        category_filter=req.category_filter
    )
    
    prod_items = [ProductItem(**p) for p in result["products"]]
    
    return AskQueryResponse(
        answer=result["answer"],
        query=result["query"],
        products=prod_items,
        grounding=GroundingInfo(**result["grounding"]),
        filters=result["filters"],
        latency=LatencyInfo(**result["latency"]),
        conversation_id=req.conversation_id
    )

@router.get("/examples")
def get_examples():
    return CURATED_EXAMPLES

@router.get("/filters")
def get_filter_options():
    catalog_service.initialize()
    return catalog_service.get_filter_options()
