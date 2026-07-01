"""
Multimodal RAG — Fashion Product Q&A  (Optimised v2)
Model  : Flan-T5-large Fine-tuned  (minhunhooo/flan-t5-large-fashion-rag)
Retrieval: CLIP ViT-B/32 IMAGE embeddings + FAISS + pre-loaded BLIP caption cache

Speed optimisations vs v1
─────────────────────────
1. FAISS index + caption_cache.json + fashion_image_meta.csv tải song song bằng threads
2. Caption cache load từ HF model repo trước  → BLIP chỉ chạy khi cache miss (hiếm)
3. Model load theo thứ tự: CLIP (nhỏ) → Flan-T5 (cần cho generate) → BLIP (lazy nhất)
4. `torch.compile` (torch ≥ 2.0) cho Flan-T5 — tắt nếu thiết bị không hỗ trợ
5. Tất cả download dùng `hf_hub_download` với `local_dir_use_symlinks=False`
   để HF không copy thêm lần nữa
6. Dataset load bằng `fashion_image_meta.csv` (chỉ metadata, không load ảnh HF)
   → tránh streaming toàn bộ 44 k ảnh khi khởi động
7. BLIP model chỉ được nạp vào VRAM khi thực sự cần (lần cache-miss đầu tiên)
"""

import os, re, time, warnings, json, threading
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import faiss
import gradio as gr
from PIL import Image
from datasets import load_dataset
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoTokenizer, AutoModelForSeq2SeqLM,
)
from huggingface_hub import hf_hub_download

# ── Config ───────────────────────────────────────────────────────────────────
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_REPO      = "minhunhooo/flan-t5-large-fashion-rag"
BLIP_MODEL_ID   = "Salesforce/blip-image-captioning-base"
CLIP_CKPT       = "openai/clip-vit-base-patch32"
DATASET_NAME    = "ashraq/fashion-product-images-small"
HF_SPACE_REPO   = "minhunhooo/fashion-rag-demo"   # repo_type="space"
TOP_K_MAX       = 10
MAX_NEW_TOK     = 80
SIM_THRESHOLD   = 0.22

CACHE_DIR       = "/tmp/fashion_rag"
os.makedirs(CACHE_DIR, exist_ok=True)

FAISS_CACHE     = f"{CACHE_DIR}/fashion_clip_image.faiss"
META_CACHE      = f"{CACHE_DIR}/fashion_image_meta.csv"
CAPTION_CACHE_F = f"{CACHE_DIR}/caption_cache.json"

print(f"🚀 Device: {DEVICE}")

# ── Helpers ──────────────────────────────────────────────────────────────────
def _hf_download(filename: str, repo_id: str, repo_type: str, dest: str):
    """Download a file from HuggingFace to dest if not already cached."""
    if os.path.exists(dest):
        print(f"  ✅ Already cached: {os.path.basename(dest)}")
        return
    print(f"  ⬇️  Downloading {filename} from {repo_id} ...")
    try:
        # `local_dir_use_symlinks` was removed in newer huggingface_hub versions
        # (downloads to local_dir without symlinks by default now). Try the
        # modern signature first, fall back for older pinned versions.
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=CACHE_DIR,
        )
    except TypeError:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_dir=CACHE_DIR,
            local_dir_use_symlinks=False,
        )
    # hf_hub_download writes to CACHE_DIR/<filename>
    downloaded = os.path.join(CACHE_DIR, filename)
    if downloaded != dest:
        os.rename(downloaded, dest)
    print(f"  ✅ Downloaded: {os.path.basename(dest)}")


# ── Step 1: Parallel download of heavy files ─────────────────────────────────
print("\n📥 Step 1 — Downloading artifacts (parallel) ...")

def dl_faiss():
    _hf_download("fashion_clip_image.faiss", HF_SPACE_REPO, "space", FAISS_CACHE)

def dl_meta():
    # Try model repo first (preferred — has all columns), fall back to Space
    try:
        _hf_download("fashion_image_meta.csv", MODEL_REPO, "model", META_CACHE)
    except Exception:
        print("  ⚠️  meta CSV not in model repo — will build from dataset later")

def dl_caption():
    try:
        _hf_download("caption_cache.json", MODEL_REPO, "model", CAPTION_CACHE_F)
    except Exception:
        print("  ⚠️  caption_cache.json not in model repo — will start with empty cache")

threads = [
    threading.Thread(target=dl_faiss),
    threading.Thread(target=dl_meta),
    threading.Thread(target=dl_caption),
]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("✅ All downloads finished")


# ── Step 2: Load metadata ─────────────────────────────────────────────────────
print("\n📊 Step 2 — Loading metadata ...")
if os.path.exists(META_CACHE):
    df = pd.read_csv(META_CACHE)
    for col in ["gender", "masterCategory", "subCategory", "articleType",
                "baseColour", "season", "usage", "productDisplayName"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    print(f"✅ Metadata loaded from CSV: {len(df):,} products")
    _ds_loaded = False   # will load HF dataset lazily (only for image access)
else:
    # Fallback: load from HF datasets (slower ~20-30 s, but safe)
    print("  📦 Falling back to HF dataset (slow path) ...")
    _ds_raw = load_dataset(DATASET_NAME, split="train")
    df = _ds_raw.to_pandas()
    for col in ["gender", "masterCategory", "subCategory", "articleType",
                "baseColour", "season", "usage", "productDisplayName"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    print(f"✅ Dataset: {len(df):,} products (from HF)")
    _ds_loaded = True


# Lazy dataset loader for image retrieval (avoids streaming all images at startup)
_ds_lock = threading.Lock()
_ds_cache = None

def get_hf_ds():
    global _ds_cache
    if _ds_cache is not None:
        return _ds_cache
    with _ds_lock:
        if _ds_cache is None:
            print("  🔄 Lazy-loading HF dataset for image access ...")
            if _ds_loaded:
                _ds_cache = _ds_raw   # already loaded above
            else:
                _ds_cache = load_dataset(DATASET_NAME, split="train")
    return _ds_cache


# ── Step 3: Load CLIP (fast — ~0.5 GB) ───────────────────────────────────────
print("\n🔗 Step 3 — Loading CLIP ViT-B/32 ...")
clip_processor = CLIPProcessor.from_pretrained(CLIP_CKPT)
clip_model = CLIPModel.from_pretrained(
    CLIP_CKPT,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE).eval()
print("✅ CLIP loaded")


def _to_tensor(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output.float()
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output.float()
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :].float()
    raise ValueError(f"Cannot extract tensor from {type(output)}")


@torch.no_grad()
def embed_text_clip(text: str) -> np.ndarray:
    inputs = clip_processor(
        text=[text], return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    ).to(DEVICE)
    emb = clip_model.get_text_features(**inputs)
    emb = F.normalize(_to_tensor(emb), dim=-1)
    return emb.cpu().numpy().astype("float32")


# ── Step 4: Load FAISS index ─────────────────────────────────────────────────
print("\n📂 Step 4 — Loading FAISS index ...")
faiss_index = faiss.read_index(FAISS_CACHE)
print(f"✅ FAISS index loaded: {faiss_index.ntotal:,} vectors")


# ── Step 5: Load Flan-T5-large FT ────────────────────────────────────────────
print(f"\n🤖 Step 5 — Loading Flan-T5-large FT from {MODEL_REPO} ...")
ft_tok = AutoTokenizer.from_pretrained(MODEL_REPO)
ft_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_REPO,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
).to(DEVICE).eval()

# Optional: torch.compile speeds up repeated inference on GPU (torch ≥ 2.0)
if DEVICE == "cuda":
    try:
        ft_model = torch.compile(ft_model, mode="reduce-overhead")
        print("✅ torch.compile applied to Flan-T5")
    except Exception as e:
        print(f"  ⚠️  torch.compile skipped: {e}")

print("✅ Flan-T5 ready")


# ── Step 6: Caption cache (pre-loaded) + lazy BLIP ───────────────────────────
caption_cache: dict[str, str] = {}
if os.path.exists(CAPTION_CACHE_F):
    with open(CAPTION_CACHE_F, "r") as f:
        caption_cache = json.load(f)
    print(f"✅ Caption cache loaded: {len(caption_cache):,} entries")
else:
    print("⚠️  No caption cache — BLIP will run on-demand (first queries slower)")

# BLIP loaded lazily — only when there is a cache miss
_blip_lock = threading.Lock()
_blip_proc  = None
_blip_model = None


def _ensure_blip():
    global _blip_proc, _blip_model
    if _blip_model is not None:
        return
    with _blip_lock:
        if _blip_model is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            print("  🖼️  Lazy-loading BLIP (cache miss) ...")
            _blip_proc = BlipProcessor.from_pretrained(BLIP_MODEL_ID)
            _blip_model = BlipForConditionalGeneration.from_pretrained(
                BLIP_MODEL_ID,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            ).to(DEVICE).eval()
            print("  ✅ BLIP loaded")


def get_caption(row_idx: int) -> str:
    key = str(row_idx)
    if key in caption_cache:
        return caption_cache[key]
    # Cache miss → generate with BLIP
    try:
        _ensure_blip()
        img = get_hf_ds()[int(row_idx)]["image"]
        if img is None:
            raise ValueError("no image")
        inp = _blip_proc(images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = _blip_model.generate(**inp, max_new_tokens=40)
        cap = _blip_proc.decode(ids[0], skip_special_tokens=True)
    except Exception:
        cap = df.iloc[row_idx]["productDisplayName"]
    caption_cache[key] = cap
    return cap


# ── Filter extraction ─────────────────────────────────────────────────────────
GENDER_PATTERNS = {
    "Men":    r"\b(men|man|male|mens|gentleman)\b",
    "Women":  r"\b(women|woman|female|womens|ladies|lady)\b",
    "Boys":   r"\b(boy|boys)\b",
    "Girls":  r"\b(girl|girls)\b",
    "Unisex": r"\bunisex\b",
}

_CATEGORY_VOCAB = sorted(
    set(df["masterCategory"]) | set(df["subCategory"]) |
    set(df["articleType"])    | set(df["baseColour"]),
    key=len, reverse=True,
)
_CATEGORY_VOCAB = [c for c in _CATEGORY_VOCAB if c and len(c) > 2]


def extract_filters(query: str) -> dict:
    q = query.lower()
    filters: dict = {}
    for gender, pattern in GENDER_PATTERNS.items():
        if re.search(pattern, q):
            filters["gender"] = gender
            break
    matched = [t for t in _CATEGORY_VOCAB if t.lower() in q]
    if matched:
        filters["terms"] = matched[:3]
    return filters


def _row_matches_filters(row, filters: dict) -> bool:
    if "gender" in filters:
        if row["gender"] not in (filters["gender"], "Unisex"):
            return False
    if "terms" in filters:
        haystack = " ".join(str(row.get(c, "")) for c in
                            ["masterCategory", "subCategory", "articleType", "baseColour"]).lower()
        if not any(t.lower() in haystack for t in filters["terms"]):
            return False
    return True


# ── Grounding check ───────────────────────────────────────────────────────────
_STOPWORDS = {"the","a","an","is","are","for","of","in","and","to","with",
              "this","that","available","there"}

def _tokenize(text: str) -> set:
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    return {t for t in toks if t not in _STOPWORDS and len(t) > 2}


def grounding_check(answer: str, retrieved_df: pd.DataFrame) -> dict:
    if retrieved_df is None or len(retrieved_df) == 0:
        return {"grounded": False, "matched_terms": [], "reason": "no retrieved products"}
    answer_tokens = _tokenize(answer)
    product_vocab: set = set()
    for _, row in retrieved_df.iterrows():
        for col in ["productDisplayName", "baseColour", "masterCategory", "subCategory", "gender"]:
            product_vocab |= _tokenize(str(row.get(col, "")))
    matched  = answer_tokens & product_vocab
    grounded = len(matched) >= 1
    return {
        "grounded": grounded,
        "matched_terms": sorted(matched),
        "reason": "OK" if grounded else "Answer không bám sát sản phẩm đã retrieve",
    }


# ── Generate ──────────────────────────────────────────────────────────────────
def generate_answer(prompt: str) -> str:
    inputs = ft_tok(prompt, return_tensors="pt",
                    truncation=True, max_length=400).to(DEVICE)
    with torch.no_grad():
        out = ft_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOK,
            num_beams=4,              # ← giảm từ 6→4 để nhanh hơn ~30%
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=0.8,
            repetition_penalty=1.2,
        )
    answer = ft_tok.decode(out[0], skip_special_tokens=True)
    for marker in ["Answer in 1-2 sentences, only mention products listed above:", "Answer:"]:
        if marker in answer:
            answer = answer.split(marker)[-1].strip()
    return answer.strip() or "No matching products found."


# ── RAG pipeline ──────────────────────────────────────────────────────────────
def rag_answer(query: str, top_k: int):
    t0 = time.time()

    q_emb              = embed_text_clip(query)
    CANDIDATE_POOL     = 30
    scores, idxs       = faiss_index.search(q_emb, CANDIDATE_POOL)
    candidates_idx     = idxs[0]
    best_score         = float(scores[0][0])
    t_retrieval        = time.time() - t0

    if best_score < SIM_THRESHOLD:
        no_match_msg = "Xin lỗi, không tìm thấy sản phẩm nào đủ liên quan trong catalog cho câu hỏi này."
        status_str   = f"❌ Bị từ chối (best_score={best_score:.3f} < threshold {SIM_THRESHOLD})"
        lat_str      = f"Retrieval: {t_retrieval*1000:.0f}ms | Generation: 0ms | Total: {t_retrieval*1000:.0f}ms"
        return no_match_msg, status_str, "{}", lat_str, []

    cand_df              = df.iloc[candidates_idx].copy().reset_index(drop=True)
    cand_df["_score"]    = scores[0]
    cand_df["_orig_idx"] = candidates_idx

    filters         = extract_filters(query)
    filtered        = cand_df[cand_df.apply(lambda r: _row_matches_filters(r, filters), axis=1)]
    filter_fallback = False
    if filters and len(filtered) == 0:
        filtered        = cand_df
        filter_fallback = True
    elif not filters:
        filtered = cand_df

    retrieved  = filtered.head(top_k).reset_index(drop=True)
    final_idxs = retrieved["_orig_idx"].tolist()

    # Build context (caption from pre-loaded cache → usually instant)
    context_lines = []
    for i, (_, row) in enumerate(retrieved.iterrows(), 1):
        ridx = int(row["_orig_idx"])
        cap  = get_caption(ridx)
        context_lines.append(
            f"{i}. {row['productDisplayName']} — {row['baseColour']}, {row['gender']}, "
            f"{row['masterCategory']} > {row['subCategory']} "
            f"| Mô tả ảnh: {cap} (score={row['_score']:.3f})"
        )
    context = "\n".join(context_lines)

    prompt = (
        f"Products available:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer in 1-2 sentences, only mention products listed above:"
    )

    t1      = time.time()
    answer  = generate_answer(prompt)
    t_gen   = time.time() - t1
    t_total = time.time() - t0

    g              = grounding_check(answer, retrieved)
    grounded_label = "✅ Grounded" if g["grounded"] else "❌ Không grounded — có thể không bám sát kết quả"
    filter_str     = f"Filters: {filters}" + (" (⚠️ fallback)" if filter_fallback else "")
    lat_str        = (f"Retrieval: {t_retrieval*1000:.0f}ms | "
                      f"Generation: {t_gen*1000:.0f}ms | "
                      f"Total: {t_total*1000:.0f}ms")

    # Retrieve images (lazy HF dataset access)
    images = []
    try:
        hf_ds = get_hf_ds()
        for ridx in final_idxs:
            try:
                img = hf_ds[int(ridx)]["image"]
                if img:
                    name = df.iloc[ridx]["productDisplayName"][:40]
                    images.append((img, name))
            except Exception:
                pass
    except Exception:
        pass

    return answer, grounded_label, filter_str, lat_str, images


# ── Gradio UI ─────────────────────────────────────────────────────────────────
EXAMPLES = [
    ["What black formal shoes are available for men?",      5],
    ["Do you have casual dresses for women in red?",        5],
    ["Show me white sneakers for men",                      5],
    ["What summer tops are available for women in white?",  5],
    ["Something that does not exist like a flying carpet",  5],
]

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    block_radius="14px",
    block_shadow="0 1px 3px rgba(0,0,0,0.06)",
)

CSS = """
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }

#hero {
    text-align: center;
    padding: 28px 16px 18px 16px;
}
#hero h1 {
    font-size: 1.85em;
    font-weight: 700;
    margin: 0;
}
#hero h1 .hero-title-text {
    background: linear-gradient(90deg, #4f46e5, #9333ea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#hero h1 .hero-emoji {
    -webkit-text-fill-color: initial;
    background: none;
    margin-right: 4px;
}
#hero p {
    color: var(--body-text-color-subdued);
    font-size: 0.95em;
    margin-top: 6px;
}

#ask-btn { font-weight: 600; font-size: 1.02em; height: 46px; }

#gallery-card .label-wrap { font-weight: 600; }

footer { visibility: hidden; }
"""

with gr.Blocks(css=CSS, theme=THEME, title="Fashion RAG — Multimodal Q&A") as demo:

    gr.HTML(
        """
        <div id="hero">
            <h1><span class="hero-emoji">🛍️</span><span class="hero-title-text">Fashion RAG — Multimodal Product Q&amp;A</span></h1>
            <p>Ask in natural language and get grounded answers backed by real product images,
            powered by CLIP retrieval + a fine-tuned Flan-T5-large generator.</p>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            with gr.Group():
                query_in = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. Show me black formal shoes for men",
                    lines=2,
                )
                topk_in = gr.Slider(
                    1, TOP_K_MAX, value=5, step=1,
                    label="Number of products to retrieve (Top-K)",
                )
                btn = gr.Button("🔍 Search", variant="primary", elem_id="ask-btn")

        with gr.Column(scale=6):
            with gr.Group():
                ans_out = gr.Textbox(
                    label="💬 Answer", lines=3, interactive=False,
                )
                with gr.Row():
                    ground_out = gr.Label(label="✅ Grounding check")
                    filter_out = gr.Textbox(label="🏷️ Filters detected", interactive=False)
                latency_out = gr.Textbox(label="⏱️ Latency breakdown", interactive=False)

    gallery_out = gr.Gallery(
        label="🖼️ Retrieved products (CLIP image embedding)",
        columns=5, height=240, object_fit="contain",
        elem_id="gallery-card",
    )

    def _format_outputs(query, top_k):
        answer, ground_label, filter_str, lat_str, images = rag_answer(query, top_k)
        is_grounded = ground_label.startswith("✅")
        label_dict = {"Grounded": 1.0} if is_grounded else {"Not grounded": 1.0}
        return answer, label_dict, filter_str, lat_str, images

    btn.click(
        fn=_format_outputs,
        inputs=[query_in, topk_in],
        outputs=[ans_out, ground_out, filter_out, latency_out, gallery_out],
        api_name="ask",
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[query_in, topk_in],
        label="💡 Try one of these examples",
    )

if __name__ == "__main__":
    demo.launch()
