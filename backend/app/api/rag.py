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
        "query": "What black formal shoes are available for men?",
        "title": "Black Formal Shoes for Men",
        "category": "Men • Footwear",
        "description": "Giày tây da màu đen thanh lịch cho nam công sở"
    },
    {
        "query": "Do you have casual dresses for women in red?",
        "title": "Women Casual Red Dresses",
        "category": "Women • Apparel",
        "description": "Váy đầm dạo phố màu đỏ dành cho phái đẹp"
    },
    {
        "query": "Recommend a stylish office outfit for men",
        "title": "Men Office Outfit (Mix & Match)",
        "category": "AI Stylist • Complete Outfit",
        "description": "Gợi ý trọn bộ set đồ nam công sở (Áo, Quần, Giày, Phụ kiện)"
    },
    {
        "query": "Show me white sports sneakers for men",
        "title": "Men White Sports Sneakers",
        "category": "Men • Footwear",
        "description": "Giày sneaker thể thao màu trắng năng động"
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
        is_outfit=result.get("is_outfit", False),
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
