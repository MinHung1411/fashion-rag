from fastapi import APIRouter, HTTPException, Response
from ..services.catalog_service import catalog_service
from ..schemas.rag import ProductItem

router = APIRouter(prefix="/products", tags=["Products"])

# Clean modern SVG placeholder fallback if image is missing
PLACEHOLDER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" width="300" height="400" viewBox="0 0 300 400" fill="none">
  <rect width="300" height="400" fill="#1e293b"/>
  <circle cx="150" cy="180" r="50" fill="#334155"/>
  <path d="M125 180L145 160L175 190L165 200" stroke="#94a3b8" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="165" cy="165" r="4" fill="#94a3b8"/>
  <text x="150" y="270" font-family="system-ui, sans-serif" font-size="14" fill="#64748b" text-anchor="middle">Fashion Product</text>
</svg>"""

@router.get("/{product_id}")
def get_product(product_id: int):
    catalog_service.initialize()
    p = catalog_service.get_product_by_id(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="Sản phẩm không tồn tại trong catalog.")
    return p

@router.get("/{product_id}/image")
def get_product_image(product_id: int):
    catalog_service.initialize()
    img_bytes = catalog_service.get_image_bytes(product_id=product_id)
    if img_bytes:
        return Response(content=img_bytes, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
    return Response(content=PLACEHOLDER_SVG.encode("utf-8"), media_type="image/svg+xml")

@router.get("/by-row/{row_idx}")
def get_product_by_row(row_idx: int):
    catalog_service.initialize()
    p = catalog_service.get_product_by_row(row_idx)
    if not p:
        raise HTTPException(status_code=404, detail="Sản phẩm không tồn tại.")
    return p

@router.get("/by-row/{row_idx}/image")
def get_product_image_by_row(row_idx: int):
    catalog_service.initialize()
    img_bytes = catalog_service.get_image_bytes(row_idx=row_idx)
    if img_bytes:
        return Response(content=img_bytes, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
    return Response(content=PLACEHOLDER_SVG.encode("utf-8"), media_type="image/svg+xml")
