import os
import json
import logging
import pandas as pd
from typing import Optional, Dict, Any, List
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq
from ..core.config import settings

logger = logging.getLogger(__name__)

class CatalogService:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.caption_cache: Dict[str, str] = {}
        self.id_to_row: Dict[int, int] = {}
        self.parquet1_table = None
        self.parquet2_table = None
        self.split_offset = 22036
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        logger.info("Initializing CatalogService...")
        # 1. Load metadata CSV
        if not os.path.exists(settings.META_PATH):
            logger.info("Downloading fashion_image_meta.csv from HF...")
            hf_hub_download(
                repo_id=settings.MODEL_REPO,
                filename="fashion_image_meta.csv",
                repo_type="model",
                local_dir=settings.DATA_CACHE_DIR
            )

        self.df = pd.read_csv(settings.META_PATH)
        for col in ["gender", "masterCategory", "subCategory", "articleType",
                    "baseColour", "season", "usage", "productDisplayName"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna("").astype(str)

        for idx, row in self.df.iterrows():
            self.id_to_row[int(row["id"])] = idx

        # 2. Load caption cache
        if not os.path.exists(settings.CAPTION_PATH):
            logger.info("Downloading caption_cache.json from HF...")
            hf_hub_download(
                repo_id=settings.MODEL_REPO,
                filename="caption_cache.json",
                repo_type="model",
                local_dir=settings.DATA_CACHE_DIR
            )

        if os.path.exists(settings.CAPTION_PATH):
            with open(settings.CAPTION_PATH, "r", encoding="utf-8") as f:
                self.caption_cache = json.load(f)

        # 3. Load Parquet files for image access
        try:
            p1_path = hf_hub_download(
                repo_id=settings.DATASET_NAME,
                filename="data/train-00000-of-00002-6cff4c59f91661c3.parquet",
                repo_type="dataset"
            )
            self.parquet1_table = pq.read_table(p1_path, columns=["id", "image"])
            self.split_offset = len(self.parquet1_table)

            p2_path = hf_hub_download(
                repo_id=settings.DATASET_NAME,
                filename="data/train-00001-of-00002-bb459e5ac5f01e71.parquet",
                repo_type="dataset"
            )
            self.parquet2_table = pq.read_table(p2_path, columns=["id", "image"])
            logger.info("Parquet image tables cached successfully.")
        except Exception as e:
            logger.warning(f"Parquet lazy load notice: {e}")

        self._initialized = True
        logger.info(f"CatalogService initialized: {len(self.df)} products, {len(self.caption_cache)} captions.")

    def get_product_by_row(self, row_idx: int) -> Optional[Dict[str, Any]]:
        if self.df is None or row_idx < 0 or row_idx >= len(self.df):
            return None
        row = self.df.iloc[row_idx].to_dict()
        row["row_idx"] = int(row_idx)
        row["caption"] = self.get_caption(row_idx)
        row["image_url"] = f"/api/products/{int(row['id'])}/image"
        return row

    def get_product_by_id(self, product_id: int) -> Optional[Dict[str, Any]]:
        row_idx = self.id_to_row.get(product_id)
        if row_idx is not None:
            return self.get_product_by_row(row_idx)
        return None

    def get_caption(self, row_idx: int) -> str:
        key = str(row_idx)
        if key in self.caption_cache:
            return self.caption_cache[key]
        if self.df is not None and row_idx < len(self.df):
            return str(self.df.iloc[row_idx].get("productDisplayName", ""))
        return ""

    def get_image_bytes(self, product_id: Optional[int] = None, row_idx: Optional[int] = None) -> Optional[bytes]:
        if row_idx is None and product_id is not None:
            row_idx = self.id_to_row.get(product_id)

        if row_idx is None:
            return None

        try:
            if row_idx < self.split_offset and self.parquet1_table is not None:
                cell = self.parquet1_table["image"][row_idx].as_py()
                if isinstance(cell, dict) and "bytes" in cell:
                    return cell["bytes"]
            elif self.parquet2_table is not None:
                offset = row_idx - self.split_offset
                if offset < len(self.parquet2_table):
                    cell = self.parquet2_table["image"][offset].as_py()
                    if isinstance(cell, dict) and "bytes" in cell:
                        return cell["bytes"]
        except Exception as e:
            logger.error(f"Error fetching image bytes for row {row_idx}: {e}")

        return None

    def get_filter_options(self) -> Dict[str, List[str]]:
        if self.df is None:
            return {}
        return {
            "genders": sorted(list(set(self.df["gender"].dropna()))),
            "masterCategories": sorted(list(set(self.df["masterCategory"].dropna()))),
            "subCategories": sorted(list(set(self.df["subCategory"].dropna()))),
            "seasons": sorted(list(set(self.df["season"].dropna()))),
            "baseColours": sorted(list(set(self.df["baseColour"].dropna())))[:25]
        }

catalog_service = CatalogService()
