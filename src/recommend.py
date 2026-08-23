"""FashionCLIP / CLIP recommendation: Top-5 similar catalog items."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote_plus

import re

import numpy as np
import pandas as pd
import torch
from PIL import Image

from .hf_auth import ensure_hf_login, hf_token

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_CSV = ROOT / "data" / "catalog.csv"
DEFAULT_EMBEDDINGS = ROOT / "data" / "embeddings.npy"

FASHION_CLIP_ID = "Marqo/marqo-fashionCLIP"
FALLBACK_CLIP_ID = "openai/clip-vit-base-patch32"

COLOR_SYNONYMS = {
    "navy": ["navy", "navy blue", "soft navy", "blue"],
    "navy blue": ["navy", "navy blue", "soft navy", "blue"],
    "soft navy": ["navy", "navy blue", "blue"],
    "rose": ["rose", "pink", "mauve", "dusty rose"],
    "dusty rose": ["rose", "pink", "mauve"],
    "mauve": ["mauve", "purple", "pink", "rose", "violet"],
    "sage": ["sage", "green", "olive", "leaf green"],
    "taupe": ["taupe", "beige", "cream", "khaki", "camel", "brown", "tan"],
    "lavender": ["lavender", "purple", "violet", "lilac"],
    "powder blue": ["powder blue", "blue", "sky"],
    "grey": ["grey", "gray", "charcoal", "soft grey"],
    "gray": ["grey", "gray", "charcoal", "soft grey"],
    "soft grey": ["grey", "gray", "charcoal"],
    "cream": ["cream", "ivory", "beige", "white"],
    "ivory": ["ivory", "cream", "white"],
    "black": ["black", "charcoal"],
    "olive": ["olive", "green", "sage"],
    "camel": ["camel", "beige", "taupe", "tan", "khaki"],
}

_WOMAN_RE = re.compile(r"\b(women|woman|womens|ladies|lady|girls?|girl's)\b", re.I)
_MAN_RE = re.compile(r"\b(men|mens|man's|male|boys?|boy's)\b", re.I)
_KIDS_RE = re.compile(r"\b(kids?|kid's|teen|girls?|boys?|boy's|girl's)\b", re.I)
_WOMAN_GARMENT_RE = re.compile(
    r"\b(dress|skirt|gown|kurti|jumpsuit|blouse|legging|leggings)\b", re.I
)


def _as_vector(feats: torch.Tensor) -> np.ndarray:
    if feats.ndim > 1:
        feats = feats[0]
    feats = feats.detach().float().cpu()
    feats = feats / (feats.norm() + 1e-8)
    return feats.numpy().astype(np.float32).reshape(-1)


@lru_cache(maxsize=1)
def load_clip():
    """
    Load Marqo FashionCLIP via open_clip; fall back to OpenAI CLIP ViT-B/32.
    Returns (encode_image_fn, encode_text_fn, device, model_name)
    """
    ensure_hf_login()
    token = hf_token()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Preferred: open_clip hub load for Marqo FashionCLIP
    try:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            f"hf-hub:{FASHION_CLIP_ID}"
        )
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(f"hf-hub:{FASHION_CLIP_ID}")

        def encode_image(img: Image.Image) -> np.ndarray:
            if img.mode != "RGB":
                img = img.convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feats = model.encode_image(tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return _as_vector(feats)

        def encode_text(text: str) -> np.ndarray:
            tokens = tokenizer([text]).to(device)
            with torch.no_grad():
                feats = model.encode_text(tokens)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return _as_vector(feats)

        return encode_image, encode_text, device, FASHION_CLIP_ID
    except Exception:
        pass

    # 2) Fallback: transformers OpenAI CLIP
    from transformers import CLIPModel, CLIPProcessor

    kw = {"token": token} if token else {}
    model = CLIPModel.from_pretrained(FALLBACK_CLIP_ID, **kw)
    processor = CLIPProcessor.from_pretrained(FALLBACK_CLIP_ID, **kw)
    model.to(device)
    model.eval()

    def encode_image(img: Image.Image) -> np.ndarray:
        if img.mode != "RGB":
            img = img.convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return _as_vector(feats)

    def encode_text(text: str) -> np.ndarray:
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return _as_vector(feats)

    return encode_image, encode_text, device, FALLBACK_CLIP_ID


def embed_image(image: Image.Image) -> np.ndarray:
    encode_image, _, _, _ = load_clip()
    return encode_image(image)


def embed_text(text: str) -> Optional[np.ndarray]:
    try:
        _, encode_text, _, _ = load_clip()
        if encode_text is None:
            return None
        return encode_text(text)
    except Exception:
        return None


def catalog_audience(title: str, category: str = "") -> str:
    """menswear / womenswear / unisex from catalog title and category."""
    blob = f"{title} {category}"
    woman = bool(_WOMAN_RE.search(blob))
    man = bool(_MAN_RE.search(blob))
    if woman and not man:
        return "woman"
    if man and not woman:
        return "man"
    if str(category or "").strip().lower() == "dress" or _WOMAN_GARMENT_RE.search(str(title or "")):
        return "woman"
    cat = str(category or "").strip().lower()
    if cat in {"shoes", "shoe"} and re.search(
        r"\b(heel|heels|pump|stiletto|sandal|mule)\b", str(title or ""), re.I
    ):
        return "woman"
    return "unisex"


def is_kidswear(title: str) -> bool:
    return bool(_KIDS_RE.search(str(title or "")))


def infer_clothing_audience(image: Image.Image) -> str:
    """Zero-shot CLIP: man vs woman from the uploaded photo."""
    try:
        img = embed_image(image)
        img = img / (np.linalg.norm(img) + 1e-8)
        man_prompts = (
            "a photograph of a man",
            "a male model wearing menswear",
            "menswear outfit on a man",
        )
        woman_prompts = (
            "a photograph of a woman",
            "a female model wearing womenswear",
            "womenswear outfit on a woman",
        )

        def _best(prompts: Sequence[str]) -> float:
            best = -1.0
            for text in prompts:
                vec = embed_text(text)
                if vec is None:
                    continue
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                best = max(best, float(np.dot(img, vec)))
            return best

        man = _best(man_prompts)
        woman = _best(woman_prompts)
        if man < 0 and woman < 0:
            return "unisex"
        if man - woman > 0.008:
            return "man"
        if woman - man > 0.008:
            return "woman"
    except Exception:
        pass
    return "unisex"


def _audience_ok(title: str, category: str, want: Optional[str]) -> bool:
    key = str(want or "").strip().lower()
    if key in {"", "neutral", "unisex", "auto"}:
        return True
    if is_kidswear(title):
        return False
    got = catalog_audience(title, category)
    if key in {"man", "male", "men", "menswear"}:
        return got in {"man", "unisex"}
    if key in {"woman", "female", "women", "womenswear"}:
        return got in {"woman", "unisex"}
    return True


def shop_url_for(title: str, existing: Optional[str] = None) -> str:
    if existing and str(existing).startswith("http"):
        return str(existing)
    q = quote_plus(f"{title} buy online")
    return f"https://www.google.com/search?tbm=shop&q={q}"


def load_catalog(
    csv_path: Path = DEFAULT_CATALOG_CSV,
    emb_path: Path = DEFAULT_EMBEDDINGS,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Catalog not found at {csv_path}. Run: python -m src.catalog_builder"
        )
    df = pd.read_csv(csv_path)
    emb = np.load(emb_path) if emb_path.exists() else None
    return df, emb


def ensure_embeddings(
    df: pd.DataFrame,
    emb: Optional[np.ndarray],
    emb_path: Path = DEFAULT_EMBEDDINGS,
) -> np.ndarray:
    """Compute embeddings if missing or shape mismatch."""
    if emb is not None and len(emb) == len(df):
        return emb

    vectors = []
    for _, row in df.iterrows():
        path = ROOT / str(row["image_path"])
        if not path.exists():
            path = Path(str(row["image_path"]))
        img = Image.open(path).convert("RGB")
        vectors.append(embed_image(img))
    arr = np.stack(vectors, axis=0).astype(np.float32)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, arr)
    return arr


def _color_boost(row_color: str, hints: Sequence[str], weight: float = 0.08) -> float:
    color = str(row_color or "").lower()
    if not color or not hints:
        return 0.0
    color_bits = set(color.replace("-", " ").split())
    color_bits.add(color)
    boost = 0.0
    for hint in hints:
        h = str(hint).lower().strip()
        if not h:
            continue
        syns = COLOR_SYNONYMS.get(h, [h])
        tokens = {h, *syns, *h.split()}
        if any(tok in color or color in tok or tok in color_bits for tok in tokens if len(tok) > 2):
            boost = max(boost, weight)
        elif any(part in color for part in h.split() if len(part) > 2):
            boost = max(boost, weight * 0.5)
    return boost


def _as_result_row(row, idx: int, sim: float) -> Dict[str, Any]:
    title = str(row.get("title", f"Item {idx}"))
    return {
        "id": row.get("id", idx),
        "title": title,
        "category": row.get("category", ""),
        "color": row.get("color", ""),
        "image_path": str(row.get("image_path", "")),
        "similarity": sim,
        "shop_url": shop_url_for(title, row.get("shop_url")),
        "high_confidence": sim >= 0.85,
    }


def _rank_catalog(
    sims: np.ndarray,
    df: pd.DataFrame,
    k: int,
    min_sim: float,
    exclude_ids: Optional[Sequence],
    offset: int,
    color_hints: Optional[Sequence[str]],
    color_weight: float = 0.08,
    avoid_colors: Optional[Sequence[str]] = None,
    audience: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    scores = sims.copy()
    want = str(audience or "").strip().lower()
    allow_cats = {str(c).strip().lower() for c in (categories or []) if str(c).strip()}
    if color_hints:
        for i, row in df.iterrows():
            scores[int(i)] = float(scores[int(i)]) + _color_boost(
                row.get("color", ""), color_hints, weight=color_weight
            )
    if avoid_colors:
        avoid_bits = {str(a).lower().strip() for a in avoid_colors if str(a).strip()}
        for i, row in df.iterrows():
            color = str(row.get("color") or "").lower()
            if any(a in color or color in a for a in avoid_bits if len(a) > 2):
                scores[int(i)] = float(scores[int(i)]) - color_weight
    if want in {"man", "male", "men", "menswear", "woman", "female", "women", "womenswear"}:
        target = "man" if want in {"man", "male", "men", "menswear"} else "woman"
        for i, row in df.iterrows():
            got = catalog_audience(str(row.get("title", "")), str(row.get("category", "")))
            if got == target:
                scores[int(i)] = float(scores[int(i)]) + 0.05
    skip = {str(x) for x in (exclude_ids or [])}
    order = np.argsort(-scores)
    results: List[Dict[str, Any]] = []
    skipped = 0
    for idx in order:
        row = df.iloc[int(idx)]
        item_id = str(row.get("id", idx))
        if item_id in skip:
            continue
        if not _audience_ok(str(row.get("title", "")), str(row.get("category", "")), want):
            continue
        if allow_cats and str(row.get("category", "")).strip().lower() not in allow_cats:
            continue
        sim = float(scores[int(idx)])
        if float(sims[int(idx)]) < min_sim:
            continue
        if skipped < offset:
            skipped += 1
            continue
        results.append(_as_result_row(row, int(idx), sim))
        if len(results) >= k:
            break
    return results


def recommend_top_k(
    query_image: Image.Image,
    k: int = 5,
    csv_path: Path = DEFAULT_CATALOG_CSV,
    emb_path: Path = DEFAULT_EMBEDDINGS,
    min_sim: float = 0.0,
    exclude_ids: Optional[Sequence] = None,
    offset: int = 0,
    color_hints: Optional[Sequence[str]] = None,
    color_weight: float = 0.08,
    avoid_colors: Optional[Sequence[str]] = None,
    audience: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Return Top-k catalog items with cosine similarity confidence."""
    df, emb = load_catalog(csv_path, emb_path)
    emb = ensure_embeddings(df, emb, emb_path)

    q = embed_image(query_image)
    q = q / (np.linalg.norm(q) + 1e-8)
    sims = emb @ q
    return _rank_catalog(
        sims,
        df,
        k,
        min_sim,
        exclude_ids,
        offset,
        color_hints,
        color_weight,
        avoid_colors,
        audience=audience,
        categories=categories,
    )


def recommend_from_text(
    query: str,
    k: int = 5,
    csv_path: Path = DEFAULT_CATALOG_CSV,
    emb_path: Path = DEFAULT_EMBEDDINGS,
    exclude_ids: Optional[Sequence] = None,
    color_hints: Optional[Sequence[str]] = None,
    color_weight: float = 0.25,
    avoid_colors: Optional[Sequence[str]] = None,
    audience: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Rank catalog items from a natural-language style request."""
    df, emb = load_catalog(csv_path, emb_path)
    emb = ensure_embeddings(df, emb, emb_path)
    allow_cats = {str(c).strip().lower() for c in (categories or []) if str(c).strip()}
    q = embed_text(query)
    if q is not None:
        q = q / (np.linalg.norm(q) + 1e-8)
        sims = emb @ q
        return _rank_catalog(
            sims,
            df,
            k,
            0.0,
            exclude_ids,
            0,
            color_hints,
            color_weight,
            avoid_colors,
            audience=audience,
            categories=categories,
        )

    skip = {str(x) for x in (exclude_ids or [])}
    tokens = [t for t in query.lower().replace(",", " ").split() if len(t) > 2]
    scored: List[tuple[float, int]] = []
    for i, row in df.iterrows():
        item_id = str(row.get("id", i))
        if item_id in skip:
            continue
        if allow_cats and str(row.get("category", "")).strip().lower() not in allow_cats:
            continue
        if not _audience_ok(str(row.get("title", "")), str(row.get("category", "")), audience):
            continue
        blob = f"{row.get('title', '')} {row.get('color', '')} {row.get('category', '')}".lower()
        hits = sum(1 for t in tokens if t in blob)
        scored.append((float(hits), int(i)))
    scored.sort(key=lambda x: -x[0])
    results = []
    for hits, idx in scored[:k]:
        row = df.iloc[idx]
        sim = min(0.92, 0.45 + 0.08 * hits)
        results.append(_as_result_row(row, idx, sim))
    return results


def clip_similarity(image_a: Image.Image, image_b: Image.Image) -> float:
    """Cosine similarity between two images in CLIP space."""
    a = embed_image(image_a)
    b = embed_image(image_b)
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def crop_by_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Crop image to mask bounding box for CLIP comparison."""
    m = np.asarray(mask.convert("L"))
    ys, xs = np.where(m > 127)
    if len(xs) == 0:
        return image
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return image.crop((x0, y0, x1, y1))
