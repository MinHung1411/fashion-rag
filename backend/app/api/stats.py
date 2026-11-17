from fastapi import APIRouter
from ..services.rag_engine import rag_engine
from ..services.catalog_service import catalog_service
from ..core.config import settings
from ..schemas.rag import SystemStatsResponse

router = APIRouter(prefix="/stats", tags=["System Stats"])

@router.get("", response_model=SystemStatsResponse)
def get_system_stats():
    total_vectors = rag_engine.faiss_index.ntotal if rag_engine.faiss_index else 0
    total_products = len(catalog_service.df) if catalog_service.df is not None else 0
    caption_cache_size = len(catalog_service.caption_cache)
    
    return SystemStatsResponse(
        device=settings.DEVICE.upper(),
        total_vectors=total_vectors,
        total_products=total_products,
        caption_cache_size=caption_cache_size,
        clip_loaded=rag_engine.clip_model is not None,
        t5_loaded=rag_engine.t5_ready,
        faiss_loaded=rag_engine.faiss_index is not None,
        dataset_loaded=catalog_service.parquet1_table is not None,
        status="Operational" if total_vectors > 0 else "Initializing"
    )
