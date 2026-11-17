import os
import re
import time
import logging
import threading
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
from ..core.config import settings
from .catalog_service import catalog_service

logger = logging.getLogger(__name__)

GENDER_PATTERNS = {
    "Men": r"\b(men|man|male|mens|gentleman|gentlemen)\b",
    "Women": r"\b(women|woman|female|womens|ladies|lady)\b",
    "Boys": r"\b(boy|boys)\b",
    "Girls": r"\b(girl|girls)\b",
    "Unisex": r"\bunisex\b",
}

STOPWORDS = {
    "the", "a", "an", "is", "are", "for", "of", "in", "and", "to", "with",
    "this", "that", "available", "there", "show", "me", "what", "find",
    "recommend", "looking", "have", "you", "products", "item", "items"
}

def _to_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.float()
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output.float()
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :].float()
    raise ValueError(f"Cannot extract tensor from {type(output)}")

def tokenize(text: str) -> set:
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) > 2}

class RagEngine:
    def __init__(self):
        self.device = settings.DEVICE
        self.faiss_index: Optional[faiss.Index] = None
        self.clip_processor: Optional[CLIPProcessor] = None
        self.clip_model: Optional[CLIPModel] = None
        
        self.ft_tok: Optional[AutoTokenizer] = None
        self.ft_model: Optional[AutoModelForSeq2SeqLM] = None
        self.t5_ready: bool = False
        self.t5_loading: bool = False
        
        self._category_vocab: List[str] = []
        self._lock = threading.Lock()
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        logger.info(f"Initializing RAG Engine on device: {self.device}")
        
        # 1. Ensure catalog is ready
        catalog_service.initialize()
        
        # 2. Build category vocabulary for filter extraction
        df = catalog_service.df
        if df is not None:
            vocab = set(df["masterCategory"]) | set(df["subCategory"]) | set(df["articleType"]) | set(df["baseColour"])
            self._category_vocab = sorted([c for c in vocab if c and len(c) > 2], key=len, reverse=True)

        # 3. Load FAISS index
        if not os.path.exists(settings.FAISS_PATH):
            from huggingface_hub import hf_hub_download
            logger.info("Downloading FAISS index from HF...")
            hf_hub_download(
                repo_id=settings.HF_SPACE_REPO,
                filename="fashion_clip_image.faiss",
                repo_type="space",
                local_dir=settings.DATA_CACHE_DIR
            )
            
        logger.info(f"Loading FAISS index from {settings.FAISS_PATH}...")
        try:
            with open(settings.FAISS_PATH, "rb") as f:
                index_bytes = f.read()
            arr = np.frombuffer(index_bytes, dtype=np.uint8)
            self.faiss_index = faiss.deserialize_index(arr)
        except Exception as e:
            logger.warning(f"deserialize_index notice: {e}, attempting direct read_index...")
            self.faiss_index = faiss.read_index(settings.FAISS_PATH)
        logger.info(f"FAISS index loaded with {self.faiss_index.ntotal} vectors.")

        # 4. Load CLIP model
        logger.info(f"Loading CLIP model: {settings.CLIP_CKPT}...")
        self.clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_CKPT)
        self.clip_model = CLIPModel.from_pretrained(
            settings.CLIP_CKPT,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        logger.info("CLIP loaded successfully.")

        # 5. Start background loading for Flan-T5 model
        self._start_async_t5_loader()

        self._initialized = True
        logger.info("RAG Engine core initialized.")

    def _start_async_t5_loader(self):
        def _loader():
            if self.t5_loading or self.t5_ready:
                return
            self.t5_loading = True
            try:
                logger.info(f"Async loading Flan-T5 from {settings.MODEL_REPO}...")
                self.ft_tok = AutoTokenizer.from_pretrained(settings.MODEL_REPO)
                self.ft_model = AutoModelForSeq2SeqLM.from_pretrained(
                    settings.MODEL_REPO,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                ).to(self.device).eval()
                self.t5_ready = True
                logger.info("Flan-T5-large fine-tuned model loaded & active.")
            except Exception as e:
                logger.warning(f"Notice during async T5 load: {e}. Smart generative fallback will remain active.")
            finally:
                self.t5_loading = False

        thread = threading.Thread(target=_loader, daemon=True)
        thread.start()

    @torch.no_grad()
    def embed_text_clip(self, text: str) -> np.ndarray:
        if self.clip_processor is None or self.clip_model is None:
            raise RuntimeError("CLIP model is not initialized.")
            
        inputs = self.clip_processor(
            text=[text], return_tensors="pt",
            padding=True, truncation=True, max_length=77
        ).to(self.device)
        
        output = self.clip_model.get_text_features(**inputs)
        emb = _to_tensor(output)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype("float32")

    def extract_filters(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        filters: Dict[str, Any] = {}
        for gender, pattern in GENDER_PATTERNS.items():
            if re.search(pattern, q):
                filters["gender"] = gender
                break

        matched_terms = [t for t in self._category_vocab if t.lower() in q]
        if matched_terms:
            filters["terms"] = matched_terms[:3]
        return filters

    def _row_matches_filters(self, row: dict, filters: dict) -> bool:
        if "gender" in filters:
            if row.get("gender") not in (filters["gender"], "Unisex"):
                return False
        if "terms" in filters:
            haystack = " ".join(str(row.get(c, "")) for c in
                                ["masterCategory", "subCategory", "articleType", "baseColour"]).lower()
            if not any(t.lower() in haystack for t in filters["terms"]):
                return False
        return True

    def grounding_check(self, answer: str, retrieved_products: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not retrieved_products:
            return {"grounded": False, "status": "none", "matched_terms": [], "reason": "No retrieved products"}
        
        answer_tokens = tokenize(answer)
        product_vocab: set = set()
        for prod in retrieved_products:
            for col in ["productDisplayName", "baseColour", "masterCategory", "subCategory", "gender", "articleType"]:
                product_vocab |= tokenize(str(prod.get(col, "")))
        
        matched = answer_tokens & product_vocab
        grounded = len(matched) >= 1
        return {
            "grounded": grounded,
            "status": "grounded" if grounded else "not_grounded",
            "matched_terms": sorted(list(matched)),
            "reason": "Grounded to retrieved catalog context" if grounded else "Answer not strictly grounded to retrieved items"
        }

    def _smart_fallback_generation(self, query: str, products: List[Dict[str, Any]]) -> str:
        """High quality deterministic context synthesizer when T5 weights are loading."""
        if not products:
            return "Xin lỗi, không tìm thấy sản phẩm thời trang nào phù hợp với yêu cầu của bạn."
            
        top1 = products[0]
        top1_name = top1.get("productDisplayName", "sản phẩm thời trang")
        color = top1.get("baseColour", "")
        category = top1.get("articleType") or top1.get("subCategory", "")
        gender = top1.get("gender", "")
        
        if len(products) == 1:
            return f"Dựa trên tìm kiếm, sản phẩm phù hợp nhất là '{top1_name}' ({color} {category} dành cho {gender})."
        
        top2 = products[1]
        top2_name = top2.get("productDisplayName", "")
        return (
            f"Các lựa chọn hàng đầu gồm có '{top1_name}' và '{top2_name}', "
            f"đáp ứng hoàn hảo tiêu chí '{query}' với thiết kế {color.lower()} và phong cách thời trang chuẩn xu hướng."
        )

    def generate_answer(self, prompt: str, query: str, products: List[Dict[str, Any]]) -> str:
        if self.t5_ready and self.ft_model is not None and self.ft_tok is not None:
            try:
                inputs = self.ft_tok(prompt, return_tensors="pt", truncation=True, max_length=400).to(self.device)
                with torch.no_grad():
                    out = self.ft_model.generate(
                        **inputs,
                        max_new_tokens=settings.MAX_NEW_TOKENS,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        length_penalty=0.8,
                        repetition_penalty=1.2,
                    )
                ans = self.ft_tok.decode(out[0], skip_special_tokens=True)
                for marker in ["Answer in 1-2 sentences, only mention products listed above:", "Answer:"]:
                    if marker in ans:
                        ans = ans.split(marker)[-1].strip()
                ans = ans.strip()
                if ans and ans != "No matching products found.":
                    return ans
            except Exception as e:
                logger.warning(f"Inference warning on T5: {e}")

        return self._smart_fallback_generation(query, products)

    def ask(self, query: str, top_k: int = 5, gender_filter: Optional[str] = None, category_filter: Optional[str] = None) -> Dict[str, Any]:
        self.initialize()
        t0 = time.time()
        
        # 1. CLIP Embedding
        q_emb = self.embed_text_clip(query)
        
        # 2. FAISS Candidate Search
        scores, idxs = self.faiss_index.search(q_emb, settings.CANDIDATE_POOL)
        candidates_idx = idxs[0].tolist()
        candidate_scores = scores[0].tolist()
        best_score = float(candidate_scores[0])
        t_retrieval = time.time() - t0

        # Reject if similarity too low
        if best_score < settings.SIM_THRESHOLD:
            return {
                "answer": "Xin lỗi, không tìm thấy sản phẩm nào đủ độ liên quan trong catalog 44,000+ sản phẩm cho câu hỏi này.",
                "query": query,
                "products": [],
                "grounding": {
                    "grounded": False,
                    "status": "rejected",
                    "matched_terms": [],
                    "reason": f"Độ tương đồng cao nhất ({best_score:.3f}) thấp hơn ngưỡng ({settings.SIM_THRESHOLD})"
                },
                "filters": {},
                "latency": {
                    "retrieval_ms": round(t_retrieval * 1000, 1),
                    "generation_ms": 0.0,
                    "total_ms": round(t_retrieval * 1000, 1)
                }
            }

        # 3. Filter Extraction & Filtering
        filters = self.extract_filters(query)
        if gender_filter:
            filters["gender"] = gender_filter
        if category_filter:
            if "terms" not in filters:
                filters["terms"] = []
            filters["terms"].append(category_filter)

        candidate_items = []
        for ridx, score in zip(candidates_idx, candidate_scores):
            p = catalog_service.get_product_by_row(ridx)
            if p:
                p["score"] = float(score)
                candidate_items.append(p)

        filtered_items = [p for p in candidate_items if self._row_matches_filters(p, filters)]
        filter_fallback = False
        if filters and len(filtered_items) == 0:
            filtered_items = candidate_items
            filter_fallback = True
        elif not filters:
            filtered_items = candidate_items

        top_products = filtered_items[:top_k]

        # 4. Formulate Prompt with BLIP Captions
        context_lines = []
        for i, p in enumerate(top_products, 1):
            cap = p.get("caption") or p.get("productDisplayName", "")
            context_lines.append(
                f"{i}. {p['productDisplayName']} — {p.get('baseColour', '')}, {p.get('gender', '')}, "
                f"{p.get('masterCategory', '')} > {p.get('subCategory', '')} | Mô tả ảnh: {cap} (score={p['score']:.3f})"
            )
        context = "\n".join(context_lines)

        prompt = (
            f"Products available:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer in 1-2 sentences, only mention products listed above:"
        )

        # 5. Generate Answer
        t1 = time.time()
        answer = self.generate_answer(prompt, query, top_products)
        t_gen = time.time() - t1
        t_total = time.time() - t0

        # 6. Grounding Check
        grounding = self.grounding_check(answer, top_products)

        ret_filters = dict(filters)
        if filter_fallback:
            ret_filters["fallback"] = True

        return {
            "answer": answer,
            "query": query,
            "products": top_products,
            "grounding": grounding,
            "filters": ret_filters,
            "latency": {
                "retrieval_ms": round(t_retrieval * 1000, 1),
                "generation_ms": round(t_gen * 1000, 1),
                "total_ms": round(t_total * 1000, 1)
            }
        }

rag_engine = RagEngine()
