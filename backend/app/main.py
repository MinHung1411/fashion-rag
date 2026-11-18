import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .core.config import settings
from .core.database import Base, engine, SessionLocal
from .services.auth_service import seed_default_users
from .services.catalog_service import catalog_service
from .services.rag_engine import rag_engine

from .api.auth import router as auth_router
from .api.conversations import router as conversations_router
from .api.rag import router as rag_router
from .api.products import router as products_router
from .api.stats import router as stats_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FashionRAG")

# Ensure database tables exist immediately
Base.metadata.create_all(bind=engine)
try:
    from sqlalchemy import text
    with engine.connect() as _c:
        _cols = [r[1] for r in _c.execute(text("PRAGMA table_info(messages)")).fetchall()]
        if "image_url" not in _cols:
            _c.execute(text("ALTER TABLE messages ADD COLUMN image_url TEXT"))
            _c.commit()
        if "is_outfit" not in _cols:
            _c.execute(text("ALTER TABLE messages ADD COLUMN is_outfit JSON"))
            _c.commit()
except Exception as _mig_err:
    logger.warning(f"DB migration notice: {_mig_err}")

try:
    _init_db = SessionLocal()
    seed_default_users(_init_db)
    _init_db.close()
except Exception as e:
    logger.warning(f"Initial DB seed notice: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load catalog and RAG index in thread
    import threading
    def _warmup():
        try:
            logger.info("Warming up CatalogService and RagEngine...")
            catalog_service.initialize()
            rag_engine.initialize()
            logger.info("Warmup complete. Ready for instant queries.")
        except Exception as e:
            logger.warning(f"Background warmup error: {e}")

    threading.Thread(target=_warmup, daemon=True).start()

    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allow Hugging Face Spaces iframe embedding
@app.middleware("http")
async def add_frame_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "ALLOWALL"
    return response

# Mount API Routers
app.include_router(auth_router, prefix=settings.API_PREFIX)
app.include_router(conversations_router, prefix=settings.API_PREFIX)
app.include_router(rag_router, prefix=settings.API_PREFIX)
app.include_router(products_router, prefix=settings.API_PREFIX)
app.include_router(stats_router, prefix=settings.API_PREFIX)

# Static files directory for modern frontend SPA
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_spa():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Fashion RAG API is operational. Frontend index.html not yet generated."}

@app.get("/health")
def health_check():
    return {"status": "ok", "app": settings.PROJECT_NAME, "version": settings.VERSION}
