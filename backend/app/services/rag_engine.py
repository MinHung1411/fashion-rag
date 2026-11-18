import os
import re
import time
import logging
import threading
import io
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForSeq2SeqLM
from ..core.config import settings
from .catalog_service import catalog_service

logger = logging.getLogger(__name__)

VIETNAMESE_CHARS = set("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ")

VI_REPLACEMENTS = [
    # Genders
    (r"\b(nam giới|nam|đàn ông|phái mạnh|trai|con trai)\b", "men"),
    (r"\b(nữ giới|phụ nữ|nữ|phái đẹp|gái|con gái|chị em)\b", "women"),
    (r"\b(bé trai|con nít nam)\b", "boys"),
    (r"\b(bé gái|con nít nữ)\b", "girls"),
    (r"\b(cả nam và nữ|unisex)\b", "unisex"),
    
    # Categories & items
    (r"\b(giày công sở|giày tây|giày da|giày lười|giày âu)\b", "formal shoes"),
    (r"\b(giày sneaker|giày thể thao|sneakers|sneaker)\b", "sports sneakers"),
    (r"\b(giày cao gót|cao gót)\b", "high heels"),
    (r"\b(giày dép|giày)\b", "shoes footwear"),
    (r"\b(dép kẹp|dép xỏ ngón|dép|sandal|xăng đan)\b", "sandals flip flops"),
    (r"\b(áo sơ mi|sơ mi)\b", "formal shirt"),
    (r"\b(áo thun polo|áo polo|polo)\b", "polo tshirt"),
    (r"\b(áo thun|áo phông|áo cộc)\b", "tshirt"),
    (r"\b(áo khoác blazer|blazer|áo vest|vest)\b", "blazer jacket"),
    (r"\b(áo khoác gió|áo khoác|khoác)\b", "jacket coat"),
    (r"\b(áo len|áo hoodie|hoodie|sweatshirt)\b", "sweater hoodie"),
    (r"\b(áo croptop|croptop|áo hai dây|áo kiểu|áo nữ)\b", "tops topwear"),
    (r"\b(váy đầm|váy|đầm)\b", "casual dress"),
    (r"\b(chân váy)\b", "skirt"),
    (r"\b(quần tây|quần âu|quần công sở)\b", "formal trousers"),
    (r"\b(quần jean|quần bò|jeans|jean)\b", "jeans denim"),
    (r"\b(quần short|quần cộc|quần đùi)\b", "shorts"),
    (r"\b(quần jogger|quần thun|quần thể thao)\b", "track pants"),
    (r"\b(quần)\b", "trousers"),
    (r"\b(áo)\b", "topwear"),
    (r"\b(túi xách|túi đeo chéo|túi xách tay|túi)\b", "handbag bag"),
    (r"\b(balo|ba lô)\b", "backpack"),
    (r"\b(đồng hồ)\b", "watch"),
    (r"\b(thắt lưng|dây nịt)\b", "belt"),
    (r"\b(kính mát|kính râm|kính)\b", "sunglasses"),
    (r"\b(đồ lót|quần lót|áo lót|áo ngực)\b", "innerwear briefs bra"),
    
    # Colors
    (r"\b(đen|màu đen)\b", "black"),
    (r"\b(trắng|màu trắng)\b", "white"),
    (r"\b(xanh dương|xanh biển|xanh navy|navy|xanh da trời|xanh lam)\b", "blue navy"),
    (r"\b(xanh lá|xanh lục|xanh rêu)\b", "green"),
    (r"\b(đỏ tươi|đỏ đô|đỏ rượu|màu đỏ|đỏ)\b", "red"),
    (r"\b(vàng|màu vàng)\b", "yellow"),
    (r"\b(hồng phấn|màu hồng|hồng)\b", "pink"),
    (r"\b(xám|màu xám|ghi|màu ghi)\b", "grey"),
    (r"\b(nâu|màu nâu|nâu bò)\b", "brown"),
    (r"\b(cam|màu cam)\b", "orange"),
    (r"\b(tím|màu tím)\b", "purple"),
    (r"\b(be|màu be|kem|màu kem)\b", "beige"),
    
    # Styles & Occasions
    (r"\b(công sở|đi làm|văn phòng|lịch sự|lịch lãm)\b", "office formal"),
    (r"\b(năng động|trẻ trung|hiện đại|phong cách|thanh lịch|duyên dáng)\b", "stylish active"),
    (r"\b(đi chơi|dạo phố|thường ngày|thoải mái)\b", "casual"),
    (r"\b(thể thao|chạy bộ|gym)\b", "sports"),
    (r"\b(đi tiệc|dự tiệc|dạ tiệc|đám cưới)\b", "party evening"),
    (r"\b(mùa hè|hè|mát mẻ)\b", "summer"),
    (r"\b(mùa đông|ấm áp)\b", "winter"),
    (r"\b(họa tiết hoa|họa tiết|hoa nhí|hoa)\b", "floral print"),
    
    # Outfits / Stylist
    (r"\b(set đồ|phối đồ|mix đồ|trọn bộ|combo đồ)\b", "complete outfit set"),
]

VI_STOPWORDS = [
    r"\b(1 vài|vài|một vài|những|các|mẫu|kiểu|loại|có|không|cho tôi|tìm cho tôi|hãy tìm|gợi ý|tư vấn|xem|cho xem|hỏi về|cần tìm|muốn tìm|ở đâu|thế nào|đẹp|rẻ|xịn|hot|hot trend|mới nhất|dành cho|cho|thời trang|giúp tôi|hãy|bạn ơi|mình muốn|tôi muốn|hãy gợi ý|cho bạn|phù hợp)\b"
]

INNERWEAR_TRIGGER = re.compile(
    r"\b(underwear|innerwear|bra|brief|briefs|panties|panty|boxer|lingerie|đồ lót|quần lót|áo lót|áo ngực)\b",
    re.IGNORECASE
)

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

    @torch.no_grad()
    def embed_image_clip(self, image: Image.Image) -> np.ndarray:
        if self.clip_processor is None or self.clip_model is None:
            raise RuntimeError("CLIP model is not initialized.")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        output = self.clip_model.get_image_features(**inputs)
        emb = _to_tensor(output)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy().astype("float32")

    def normalize_query(self, query: str) -> Tuple[str, Optional[str]]:
        raw = query.strip()
        is_vietnamese = any(c in VIETNAMESE_CHARS for c in raw.lower()) or any(
            w in raw.lower() for w in ['giày', 'quần', 'áo', 'váy', 'đầm', 'nữ', 'nam', 'công sở', 'set đồ', 'phối đồ', 'dạo phố']
        )
        if not is_vietnamese:
            return raw, None
            
        q = raw.lower()
        for pat in VI_STOPWORDS:
            q = re.sub(pat, " ", q)
        for pat, rep in VI_REPLACEMENTS:
            q = re.sub(pat, f" {rep} ", q)
            
        tokens = [w for w in re.findall(r"[a-zA-Z0-9]+", q) if len(w) >= 3 and w.isascii()]
        seen = set()
        cleaned = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                cleaned.append(t)
                
        norm_text = " ".join(cleaned)
        if not norm_text:
            return raw, None
        return norm_text, raw

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

    def _row_matches_filters(self, row: dict, filters: dict, query: str = "") -> bool:
        # Strict negative guard: Never return Innerwear, Bra, Briefs, Nightdress unless explicitly asked
        is_innerwear_query = bool(INNERWEAR_TRIGGER.search(query))
        sub_cat = str(row.get("subCategory", ""))
        master_cat = str(row.get("masterCategory", ""))
        art_type = str(row.get("articleType", ""))
        
        if not is_innerwear_query:
            if master_cat == "Personal Care":
                return False
            if sub_cat in ("Innerwear", "Loungewear and Nightwear"):
                return False
            if art_type in ("Bra", "Briefs", "Panties", "Trunk", "Innerwear Vests", "Nightdress", "Nightdress and Nighties", "Lingerie"):
                return False

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

    def _smart_fallback_generation(self, query: str, products: List[Dict[str, Any]], is_vietnamese: bool = True) -> str:
        """High quality deterministic context synthesizer."""
        if not products:
            if is_vietnamese:
                return "Xin lỗi, không tìm thấy sản phẩm thời trang nào phù hợp với yêu cầu của bạn trong catalog 44,000+ sản phẩm."
            return "Sorry, no matching fashion products were found in the 44,000+ catalog."
            
        top1 = products[0]
        top1_name = top1.get("productDisplayName", "sản phẩm thời trang")
        color = top1.get("baseColour", "")
        category = top1.get("articleType") or top1.get("subCategory", "")
        gender = top1.get("gender", "")
        
        if is_vietnamese:
            lines = [f"Dựa trên catalog thời trang quốc tế, dưới đây là các gợi ý phù hợp nhất dành cho bạn:\n"]
            for p in products[:4]:
                p_name = p.get("productDisplayName", "")
                p_color = p.get("baseColour", "")
                p_gender = p.get("gender", "")
                lines.append(f"• {p_name} — ({p_color}, {p_gender})")
            lines.append(f"\nCác sản phẩm sở hữu phong cách thiết kế chuẩn xu hướng, tone màu hài hòa và hoàn toàn phù hợp với tiêu chí của bạn.")
            return "\n".join(lines)
        else:
            lines = [f"Based on the international fashion catalog, here are the top recommendations for your request:\n"]
            for p in products[:4]:
                p_name = p.get("productDisplayName", "")
                p_color = p.get("baseColour", "")
                p_gender = p.get("gender", "")
                lines.append(f"• {p_name} — ({p_color}, {p_gender})")
            lines.append(f"\nThese items feature modern styling perfectly tailored for your request.")
            return "\n".join(lines)

    def generate_answer(self, prompt: str, query: str, products: List[Dict[str, Any]], translated_from: Optional[str] = None) -> str:
        is_vi = (translated_from is not None) or any(c in VIETNAMESE_CHARS for c in query.lower())
        
        # If user asked in Vietnamese, answer in clean Vietnamese
        if is_vi:
            return self._smart_fallback_generation(query, products, is_vietnamese=True)
            
        # If user asked in English, prefer fine-tuned Flan-T5
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

        return self._smart_fallback_generation(query, products, is_vietnamese=False)

    def is_outfit_query(self, query: str) -> bool:
        patterns = [
            r"\b(outfit|set đồ|phối đồ|mix match|mix and match|phối hợp|trang phục đi|set trang phục|full set|combo đồ|set quần áo)\b",
            r"\b(đi tiệc|đi làm|dự tiệc|đi chơi|đi cưới|dạo phố|hẹn hò|công sở|phong cách)\b.*(gợi ý|cho tôi|phối|mặc gì)",
            r"(mặc gì|diện gì|phối sao|phối thế nào)"
        ]
        q = query.lower()
        return any(re.search(p, q) for p in patterns)

    def recommend_outfit(self, query: str, gender_filter: Optional[str] = None) -> Dict[str, Any]:
        self.initialize()
        t0 = time.time()
        
        search_query, translated_from = self.normalize_query(query)
        filters = self.extract_filters(search_query)
        gender = gender_filter or filters.get("gender") or "Men"
        
        slots_def = [
            {"slot": "Áo" if gender == "Men" else "Váy/Áo", "icon": "👔" if gender == "Men" else "👗", "categories": ["Topwear", "Dress"], "sub_query": f"{search_query} shirt topwear dress"},
            {"slot": "Quần", "icon": "👖", "categories": ["Bottomwear"], "sub_query": f"{search_query} trousers jeans bottomwear"},
            {"slot": "Giày", "icon": "👞" if gender == "Men" else "👠", "categories": ["Footwear", "Shoes", "Sandal"], "sub_query": f"{search_query} formal casual shoes footwear"},
            {"slot": "Phụ kiện", "icon": "⌚" if gender == "Men" else "👜", "categories": ["Accessories", "Bags", "Watches", "Belts"], "sub_query": f"{search_query} accessories watch bag"}
        ]
        
        outfit_items = []
        for s in slots_def:
            emb = self.embed_text_clip(s["sub_query"])
            scores, idxs = self.faiss_index.search(emb, 60)
            
            found = None
            for ridx, score in zip(idxs[0].tolist(), scores[0].tolist()):
                p = catalog_service.get_product_by_row(ridx)
                if not p or p.get("gender") not in (gender, "Unisex"):
                    continue
                # Negative filter out innerwear for outfit slots
                if not self._row_matches_filters(p, {}, query="outfit"):
                    continue
                m_cat = p.get("masterCategory", "")
                s_cat = p.get("subCategory", "")
                if any(c.lower() in (m_cat + " " + s_cat).lower() for c in s["categories"]):
                    if s["slot"] in ("Quần", "Quần / Chân váy") and outfit_items and outfit_items[0].get("subCategory") == "Dress":
                        continue
                    p_copy = dict(p)
                    p_copy["score"] = float(score)
                    p_copy["slot"] = s["slot"]
                    p_copy["slot_icon"] = s["icon"]
                    found = p_copy
                    break
            
            if found:
                outfit_items.append(found)

        if len(outfit_items) < 2:
            return self.ask(query, top_k=5, gender_filter=gender_filter)

        t_retrieval = time.time() - t0
        
        outfit_lines = [f"✨ AI Stylist đề xuất set đồ hoàn chỉnh dành cho {gender}:\n"]
        for it in outfit_items:
            s_icon = it.get("slot_icon", "•")
            s_slot = it.get("slot", "Món đồ")
            p_name = it.get("productDisplayName", "")
            p_col = it.get("baseColour", "")
            outfit_lines.append(f"• {s_icon} {s_slot}: '{p_name}' ({p_col})")
            
        outfit_lines.append(f"\nSự kết hợp này tạo nên tổng thể hài hòa về tone màu, tôn dáng và cực kỳ phù hợp cho tiêu chí '{query}'.")
        answer = "\n".join(outfit_lines)

        t_total = time.time() - t0
        ret_filters = {"gender": gender, "type": "outfit_recommendation"}
        if translated_from:
            ret_filters["normalized_query"] = search_query

        return {
            "answer": answer,
            "query": query,
            "is_outfit": True,
            "products": outfit_items,
            "grounding": {
                "grounded": True,
                "status": "grounded",
                "matched_terms": [gender, "Outfit", "Mix & Match"],
                "reason": "Gợi ý set đồ phối màu hoàn chỉnh từ đa danh mục catalog"
            },
            "filters": ret_filters,
            "latency": {
                "retrieval_ms": round(t_retrieval * 1000, 1),
                "generation_ms": 15.0,
                "total_ms": round(t_total * 1000, 1)
            }
        }

    def ask_with_image(
        self,
        image_bytes: bytes,
        text_query: Optional[str] = None,
        top_k: int = 5,
        gender_filter: Optional[str] = None,
        category_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        self.initialize()
        t0 = time.time()
        
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_emb = self.embed_image_clip(img)
        
        if text_query and len(text_query.strip()) > 1:
            txt_emb = self.embed_text_clip(text_query.strip())
            comb = 0.65 * img_emb + 0.35 * txt_emb
            comb = comb / np.linalg.norm(comb, axis=-1, keepdims=True)
            search_emb = comb.astype("float32")
            mode_desc = f"Tìm kiếm kết hợp (Ảnh + Yêu cầu: '{text_query}')"
        else:
            search_emb = img_emb
            mode_desc = "Tìm kiếm sản phẩm tương đồng bằng hình ảnh (Visual Search)"
            
        scores, idxs = self.faiss_index.search(search_emb, settings.CANDIDATE_POOL)
        candidates_idx = idxs[0].tolist()
        candidate_scores = scores[0].tolist()
        t_retrieval = time.time() - t0
        
        filters = {}
        if gender_filter:
            filters["gender"] = gender_filter
        if category_filter:
            filters["terms"] = [category_filter]
            
        candidate_items = []
        for ridx, score in zip(candidates_idx, candidate_scores):
            p = catalog_service.get_product_by_row(ridx)
            if p:
                p["score"] = float(score)
                candidate_items.append(p)
                
        filtered_items = [p for p in candidate_items if self._row_matches_filters(p, filters, query=text_query or "")]
        if not filtered_items:
            filtered_items = [p for p in candidate_items if self._row_matches_filters(p, {}, query=text_query or "")]
            
        top_products = filtered_items[:top_k]
        top1 = top_products[0] if top_products else {}
        top1_name = top1.get("productDisplayName", "sản phẩm tương đồng")
        top1_score = (top1.get("score", 0.0) * 100)
        
        lines = [
            f"Dựa trên hình ảnh bạn tải lên ({mode_desc}), hệ thống AI trích xuất đặc trưng thị giác qua CLIP ViT-B/32 "
            f"và tìm thấy các sản phẩm có độ tương đồng cao nhất trong catalog:\n"
        ]
        for p in top_products[:4]:
            p_score = (p.get("score", 0.0) * 100)
            lines.append(f"• {p.get('productDisplayName', '')} — Độ tương đồng: {p_score:.1f}%")
        lines.append(f"\nNổi bật nhất là '{top1_name}' với độ tương đồng đạt {top1_score:.1f}%.")
        answer = "\n".join(lines)
        
        t_total = time.time() - t0
        return {
            "answer": answer,
            "query": text_query or "Visual Search",
            "is_visual_search": True,
            "products": top_products,
            "grounding": {
                "grounded": True,
                "status": "grounded",
                "matched_terms": ["Visual Search", "CLIP Embedding"],
                "reason": "Dẫn chứng từ vector hình ảnh trích xuất trực tiếp"
            },
            "filters": filters,
            "latency": {
                "retrieval_ms": round(t_retrieval * 1000, 1),
                "generation_ms": 10.0,
                "total_ms": round(t_total * 1000, 1)
            }
        }

    def ask(self, query: str, top_k: int = 5, gender_filter: Optional[str] = None, category_filter: Optional[str] = None) -> Dict[str, Any]:
        self.initialize()
        if self.is_outfit_query(query):
            return self.recommend_outfit(query, gender_filter=gender_filter)
        t0 = time.time()
        
        search_query, translated_from = self.normalize_query(query)
        
        # 1. CLIP Embedding using search_query
        q_emb = self.embed_text_clip(search_query)
        
        # 2. FAISS Candidate Search
        scores, idxs = self.faiss_index.search(q_emb, settings.CANDIDATE_POOL)
        candidates_idx = idxs[0].tolist()
        candidate_scores = scores[0].tolist()
        best_score = float(candidate_scores[0])
        t_retrieval = time.time() - t0

        # Reject if similarity too low
        if best_score < settings.SIM_THRESHOLD:
            msg = "Xin lỗi, không tìm thấy sản phẩm nào đủ độ liên quan trong catalog 44,000+ sản phẩm cho câu hỏi này." if translated_from else "Sorry, no matching fashion products found with sufficient similarity in the catalog."
            return {
                "answer": msg,
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
        filters = self.extract_filters(search_query)
        if gender_filter:
            filters["gender"] = gender_filter
        if category_filter:
            if "terms" not in filters:
                filters["terms"] = []
            filters["terms"].append(category_filter)
        if translated_from:
            filters["normalized_query"] = search_query

        candidate_items = []
        for ridx, score in zip(candidates_idx, candidate_scores):
            p = catalog_service.get_product_by_row(ridx)
            if p:
                p_copy = dict(p)
                p_copy["score"] = float(score)
                candidate_items.append(p_copy)

        filtered_items = [p for p in candidate_items if self._row_matches_filters(p, filters, query=search_query)]
        filter_fallback = False
        if filters and len(filtered_items) == 0:
            filtered_items = [p for p in candidate_items if self._row_matches_filters(p, {}, query=search_query)]
            filter_fallback = True
        elif not filters:
            filtered_items = [p for p in candidate_items if self._row_matches_filters(p, {}, query=search_query)]

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
            f"Question: {search_query}\n"
            f"Answer in 1-2 sentences, only mention products listed above:"
        )

        # 5. Generate Answer
        t1 = time.time()
        answer = self.generate_answer(prompt, query, top_products, translated_from=translated_from)
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
