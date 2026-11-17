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
        "category": "Men Footwear",
        "icon": "👞",
        "description": "Tìm kiếm giày da công sở màu đen thanh lịch cho nam"
    },
    {
        "query": "Do you have casual dresses for women in red?",
        "category": "Women Dresses",
        "icon": "👗",
        "description": "Váy đầm dạo phố màu đỏ rực rỡ dành cho nữ"
    },
    {
        "query": "Show me white sneakers for men",
        "category": "Sneakers",
        "icon": "👟",
        "description": "Giày sneaker trắng thể thao năng động và trẻ trung"
    },
    {
        "query": "What summer tops are available for women in white?",
        "category": "Women Tops",
        "icon": "👚",
        "description": "Áo mùa hè màu trắng thoáng mát và thời trang cho nữ"
    },
    {
        "query": "Something that does not exist like a flying carpet",
        "category": "Out of domain test",
        "icon": "✨",
        "description": "Kiểm tra cơ chế từ chối (rejection threshold) với truy vấn không hợp lệ"
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
