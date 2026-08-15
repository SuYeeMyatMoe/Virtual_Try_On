"""
Build a fashion catalog + FashionCLIP embeddings.

Usage:
  python -m src.catalog_builder --from-deepfashion --n 90
  python -m src.catalog_builder --placeholders --n 30
"""

from __future__ import annotations

import argparse
import csv
import os
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "data" / "catalog" / "images"
CSV_PATH = ROOT / "data" / "catalog.csv"
EMB_PATH = ROOT / "data" / "embeddings.npy"

# Catalog photos must stay sharp for Studio try-on and the shop grid.
MIN_IMAGE_SIDE = 512
MAX_SAVE_SIZE = (1080, 1440)
JPEG_QUALITY = 90
VIEWER_API = "https://datasets-server.huggingface.co"
HIRES_DATASET = "PestoRosso/lamoda-fashion-product-images"

# Curated starter catalog (DeepFashion-style metadata). Images are generated
# locally so the app runs without downloading multi-GB datasets.
CATALOG_SPEC: List[Tuple[str, str, str, Tuple[int, int, int]]] = [
    ("navy_crew_tee", "Navy Crew Neck T-Shirt", "upper", (25, 55, 109)),
    ("white_oxford", "White Oxford Button Shirt", "upper", (245, 245, 245)),
    ("black_hoodie", "Black Pullover Hoodie", "upper", (20, 20, 20)),
    ("red_blouse", "Red Silk Blouse", "upper", (178, 34, 52)),
    ("beige_sweater", "Beige Knit Sweater", "upper", (196, 176, 148)),
    ("olive_jacket", "Olive Utility Jacket", "upper", (85, 107, 47)),
    ("striped_polo", "Blue Striped Polo", "upper", (70, 130, 180)),
    ("pink_cardigan", "Pink Soft Cardigan", "upper", (255, 182, 193)),
    ("charcoal_blazer", "Charcoal Blazer", "upper", (54, 54, 54)),
    ("teal_tunic", "Teal Tunic Top", "upper", (0, 128, 128)),
    ("denim_slim", "Slim Blue Jeans", "lower", (47, 86, 150)),
    ("black_chinos", "Black Chinos", "lower", (30, 30, 30)),
    ("khaki_trousers", "Khaki Trousers", "lower", (189, 166, 120)),
    ("grey_joggers", "Grey Joggers", "lower", (128, 128, 128)),
    ("navy_skirt", "Navy A-Line Skirt", "lower", (25, 40, 90)),
    ("plaid_skirt", "Plaid Mini Skirt", "lower", (120, 40, 40)),
    ("white_shorts", "White Tailored Shorts", "lower", (250, 250, 250)),
    ("brown_corduroy", "Brown Corduroy Pants", "lower", (121, 85, 61)),
    ("forest_cargos", "Forest Cargo Pants", "lower", (46, 79, 46)),
    ("cream_culottes", "Cream Culottes", "lower", (245, 240, 230)),
    ("floral_midi", "Floral Midi Dress", "dress", (220, 120, 140)),
    ("black_evening", "Black Evening Dress", "dress", (15, 15, 15)),
    ("summer_yellow", "Yellow Summer Dress", "dress", (255, 215, 0)),
    ("emerald_wrap", "Emerald Wrap Dress", "dress", (0, 128, 90)),
    ("lavender_maxi", "Lavender Maxi Dress", "dress", (180, 160, 220)),
    ("striped_shirt_dress", "Striped Shirt Dress", "dress", (90, 140, 200)),
    ("burgundy_shift", "Burgundy Shift Dress", "dress", (128, 0, 32)),
    ("coral_sundress", "Coral Sundress", "dress", (255, 127, 80)),
    ("slate_jumpsuit", "Slate Jumpsuit", "dress", (90, 100, 110)),
    ("ivory_lace", "Ivory Lace Dress", "dress", (255, 255, 240)),
]


def _font(size: int = 28):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def make_garment_image(
    title: str,
    category: str,
    color: Tuple[int, int, int],
    size: Tuple[int, int] = (512, 640),
) -> Image.Image:
    """Generate a simple product-style garment placeholder (offline-friendly)."""
    w, h = size
    img = Image.new("RGB", (w, h), (250, 250, 252))
    draw = ImageDraw.Draw(img)

    # Soft shadow panel
    draw.rounded_rectangle((60, 50, w - 60, h - 90), radius=28, fill=(235, 235, 240))

    cx, cy = w // 2, h // 2 - 20
    if category == "upper":
        # Torso rectangle + sleeves
        draw.rounded_rectangle((cx - 110, cy - 140, cx + 110, cy + 120), radius=20, fill=color)
        draw.ellipse((cx - 170, cy - 120, cx - 90, cy - 20), fill=color)
        draw.ellipse((cx + 90, cy - 120, cx + 170, cy - 20), fill=color)
        draw.ellipse((cx - 40, cy - 160, cx + 40, cy - 100), fill=(250, 250, 252))
    elif category == "lower":
        draw.rounded_rectangle((cx - 90, cy - 100, cx + 90, cy + 160), radius=18, fill=color)
        draw.rectangle((cx - 10, cy - 40, cx + 10, cy + 160), fill=(250, 250, 252))
    else:  # dress
        draw.polygon(
            [
                (cx - 40, cy - 150),
                (cx + 40, cy - 150),
                (cx + 130, cy + 160),
                (cx - 130, cy + 160),
            ],
            fill=color,
        )
        draw.ellipse((cx - 35, cy - 175, cx + 35, cy - 120), fill=(250, 250, 252))

    # Label
    font = _font(22)
    label = title if len(title) < 34 else title[:31] + "..."
    draw.text((40, h - 70), label, fill=(40, 40, 45), font=font)
    draw.text((40, h - 42), category.upper(), fill=(110, 110, 120), font=_font(16))
    return img


def build_catalog(n: int = 30) -> Path:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    specs = CATALOG_SPEC[: max(1, min(n, len(CATALOG_SPEC)))]
    rows = []

    for i, (slug, title, category, rgb) in enumerate(specs, start=1):
        rel = f"data/catalog/images/{slug}.png"
        out = ROOT / rel
        make_garment_image(title, category, rgb).save(out)
        color_name = slug.split("_")[0]
        shop = f"https://www.google.com/search?tbm=shop&q={quote_plus(title + ' buy')}"
        rows.append(
            {
                "id": f"DF{i:03d}",
                "title": title,
                "category": category,
                "color": color_name,
                "image_path": rel.replace("\\", "/"),
                "shop_url": shop,
            }
        )

    editorial = _existing_editorial_rows()
    combined = list(editorial) + rows
    return _write_catalog_rows(combined)


def build_embeddings() -> Path:
    from .recommend import embed_image

    if not CSV_PATH.exists():
        build_catalog()

    import pandas as pd

    df = pd.read_csv(CSV_PATH)
    vectors = []
    keep_idx = []
    for i, row in df.iterrows():
        path = ROOT / str(row["image_path"])
        if not path.exists():
            alt = ROOT / "assets" / "home" / Path(str(row["image_path"])).name
            path = alt if alt.exists() else path
        if not path.exists():
            print(f"Skip missing image: {row.get('image_path')}")
            continue
        img = Image.open(path).convert("RGB")
        vectors.append(embed_image(img))
        keep_idx.append(i)
        print(f"Embedded: {row['title']}")

    if not vectors:
        raise RuntimeError("No catalog images found to embed.")

    if len(keep_idx) != len(df):
        df = df.iloc[keep_idx].reset_index(drop=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"Catalog CSV trimmed to {len(df)} rows with existing images.")

    arr = np.stack(vectors, axis=0).astype(np.float32)
    # L2 normalize rows
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    arr = arr / norms
    np.save(EMB_PATH, arr)
    print(f"Saved embeddings {arr.shape} -> {EMB_PATH}")
    return EMB_PATH


DF_DATASETS = (
    "Marqo/deepfashion-inshop",
    "Marqo/deepfashion-multimodal",
)

DRESS_KW = ("dress", "gown", "jumpsuit", "romper", "maxi", "sundress", "frock")
LOWER_KW = ("pant", "jean", "skirt", "short", "trouser", "legging", "chino", "culotte", "jogger")
UPPER_KW = (
    "shirt", "tee", "t-shirt", "tshirt", "blouse", "jacket", "hoodie", "sweater",
    "coat", "blazer", "cardigan", "top", "polo", "tank", "knit", "tunic", "vest",
)
SKIP_KW = (
    "watch", "lipstick", "perfume", "fragrance", "belt", "shoe", "sandal",
    "heel", "earring", "sunglass", "wallet", "bag", "bra", "brief", "jewelry",
    "jewellery", "necklace", "flip flop", "sock", "tie", "cufflink",
    "goggle", "eyewear",
)
COLOR_WORDS = (
    "black", "white", "navy", "blue", "red", "green", "pink", "beige", "grey",
    "gray", "brown", "olive", "cream", "ivory", "yellow", "purple", "violet",
    "orange", "teal", "burgundy", "khaki", "charcoal", "gold", "silver", "coral",
)


def infer_category(text: str) -> str | None:
    t = f" {(text or '').lower()} "
    if any(f" {k} " in t or t.strip().endswith(k) for k in DRESS_KW):
        return "dress"
    if any(k in t for k in LOWER_KW):
        return "lower"
    if any(f" {k} " in t or t.strip().startswith(k) for k in (" shirt", " tee", " t-shirt", " blouse", " jacket", " hoodie", " sweater", " coat", " blazer", " cardigan", " top", " polo", " tank", " knit", " tunic", " vest")):
        return "upper"
    return None


def infer_color(text: str) -> str:
    t = (text or "").lower()
    for color in COLOR_WORDS:
        if color in t:
            return "grey" if color == "gray" else color
    return "multi"


def _row_text(row: dict) -> str:
    for key in (
        "product_display_name",
        "productDisplayName",
        "title",
        "text",
        "caption",
        "description",
        "article_type",
        "articleType",
        "category",
        "label",
    ):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _row_category(row: dict, title: str) -> str | None:
    master = str(
        row.get("master_category") or row.get("masterCategory") or ""
    ).lower()
    if master in {"footwear", "accessories", "personal care", "free items", "sporting goods"}:
        return None
    sub = " ".join(
        str(row.get(k) or "")
        for k in (
            "sub_category",
            "subCategory",
            "article_type",
            "articleType",
            "master_category",
            "masterCategory",
        )
    ).lower()
    if any(k in sub for k in ("innerwear", "loungewear", "free gift")):
        return None
    if any(k in sub for k in ("dress", "gown", "jumpsuit", "romper", "saree")):
        return "dress"
    if any(k in sub for k in ("bottom", "pant", "jean", "skirt", "short", "trouser")):
        return "lower"
    if any(k in sub for k in ("topwear", "top", "shirt", "jacket", "sweater", "kurta")):
        return "upper"
    return infer_category(title)


def _hf_headers() -> dict:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    headers = {"User-Agent": "vesture-catalog/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _download_rgb(url: str) -> Optional[Image.Image]:
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=_hf_headers(), timeout=60)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            return img
        except Exception as exc:
            last_err = exc
            if attempt == 2:
                print(f"Skip image download ({exc})")
    return None


def _row_image(row: dict):
    for key in ("image", "img", "product_image", "thumbnail", "jpg"):
        if key in row and row[key] is not None:
            val = row[key]
            if isinstance(val, Image.Image):
                return val.convert("RGB")
            if isinstance(val, dict) and "bytes" in val:
                return Image.open(BytesIO(val["bytes"])).convert("RGB")
            if isinstance(val, dict) and val.get("path"):
                return Image.open(val["path"]).convert("RGB")
            if isinstance(val, dict) and val.get("src"):
                return _download_rgb(str(val["src"]))
    return None


def _image_meets_min_size(row: dict, img: Optional[Image.Image] = None) -> bool:
    w = row.get("width")
    h = row.get("height")
    try:
        if w and h and min(int(w), int(h)) >= MIN_IMAGE_SIDE:
            return True
        if w and h and min(int(w), int(h)) < MIN_IMAGE_SIDE:
            return False
    except (TypeError, ValueError):
        pass
    if img is None:
        return True
    return min(img.size) >= MIN_IMAGE_SIDE


def _save_catalog_jpeg(img: Image.Image, path: Path) -> bool:
    if min(img.size) < MIN_IMAGE_SIDE:
        return False
    rgb = img.convert("RGB")
    rgb.thumbnail(MAX_SAVE_SIZE, Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return True


def _existing_editorial_rows() -> List[dict]:
    """Keep VE* lookbook rows already in catalog.csv."""
    if not CSV_PATH.exists():
        return []
    import pandas as pd

    df = pd.read_csv(CSV_PATH)
    if "id" not in df.columns:
        return []
    keep = df[df["id"].astype(str).str.startswith("VE")]
    return keep.to_dict("records")


def _write_catalog_rows(rows: List[dict]) -> Path:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "title", "category", "color", "image_path", "shop_url"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {k: row.get(k, "") for k in ["id", "title", "category", "color", "image_path", "shop_url"]}
            )
    print(f"Wrote {len(rows)} catalog items -> {CSV_PATH}")
    return CSV_PATH


def _row_blob(row: dict) -> str:
    return " ".join(
        str(row.get(k) or "")
        for k in (
            "product_display_name",
            "productDisplayName",
            "sub_category",
            "subCategory",
            "article_type",
            "articleType",
            "master_category",
            "masterCategory",
            "title",
            "text",
        )
    ).lower()


def _row_color(row: dict, title: str) -> str:
    color_from_row = str(row.get("base_color") or row.get("baseColour") or "").strip().lower()
    return color_from_row if color_from_row else infer_color(title)


def _iter_viewer_rows(dataset: str, *, length: int = 100, max_rows: int = 8000) -> Iterable[dict]:
    offset = 0
    while offset < max_rows:
        resp = requests.get(
            f"{VIEWER_API}/rows",
            params={
                "dataset": dataset,
                "config": "default",
                "split": "train",
                "offset": offset,
                "length": length,
            },
            headers=_hf_headers(),
            timeout=60,
        )
        resp.raise_for_status()
        payload = resp.json()
        page = payload.get("rows") or []
        if not page:
            break
        for item in page:
            yield item.get("row") or {}
        offset += len(page)
        if len(page) < length:
            break


def _collect_from_viewer(per_cat: int) -> List[Tuple[str, str, Image.Image, str]]:
    buckets: Dict[str, list] = {"upper": [], "lower": [], "dress": []}
    scanned = 0
    print(f"Fetching high-res catalog stills from {HIRES_DATASET} (min {MIN_IMAGE_SIDE}px)…")
    for row in _iter_viewer_rows(HIRES_DATASET, max_rows=9000):
        scanned += 1
        if all(len(buckets[c]) >= per_cat for c in buckets):
            break
        if scanned % 500 == 0:
            print(
                f"  scanned {scanned} · upper {len(buckets['upper'])}/{per_cat} · "
                f"lower {len(buckets['lower'])}/{per_cat} · dress {len(buckets['dress'])}/{per_cat}"
            )
        blob = _row_blob(row)
        if any(k in blob for k in SKIP_KW):
            continue
        title = _row_text(row) or "Fashion item"
        category = _row_category(row, title)
        if category is None or len(buckets[category]) >= per_cat:
            continue
        if not _image_meets_min_size(row):
            continue
        img = _row_image(row)
        if img is None or min(img.size) < MIN_IMAGE_SIDE:
            continue
        buckets[category].append((title, category, img, _row_color(row, title)))
        print(f"  + {category:5} {img.size[0]}x{img.size[1]}  {title[:60]}")
    print(f"High-res viewer collected {sum(len(v) for v in buckets.values())} items after {scanned} rows")
    return buckets["upper"] + buckets["lower"] + buckets["dress"]


def _collect_from_streaming(per_cat: int) -> Tuple[List[Tuple[str, str, Image.Image, str]], Exception | None]:
    from datasets import load_dataset

    buckets: Dict[str, list] = {"upper": [], "lower": [], "dress": []}
    last_err: Exception | None = None
    for repo in DF_DATASETS:
        try:
            ds = load_dataset(repo, split="train", streaming=True)
        except Exception as exc:
            last_err = exc
            continue
        scanned = 0
        for row in ds:
            scanned += 1
            if scanned > 12000 or all(len(buckets[c]) >= per_cat for c in buckets):
                break
            row = dict(row)
            title = _row_text(row) or "Fashion item"
            if any(k in _row_blob(row) for k in SKIP_KW):
                continue
            category = _row_category(row, title)
            if category is None or len(buckets[category]) >= per_cat:
                continue
            img = _row_image(row)
            if img is None or min(img.size) < MIN_IMAGE_SIDE:
                continue
            buckets[category].append((title, category, img, _row_color(row, title)))
        if any(buckets.values()):
            break
    return buckets["upper"] + buckets["lower"] + buckets["dress"], last_err


def _write_collected_items(collected: List[Tuple[str, str, Image.Image, str]]) -> Path:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for stale in IMAGES_DIR.glob("df_*.jpg"):
        stale.unlink(missing_ok=True)
    editorial = _existing_editorial_rows()
    rows = list(editorial)
    kept = 0
    for title, category, img, color_from_row in collected:
        kept += 1
        slug = f"df_{category}_{kept:03d}"
        rel = f"data/catalog/images/{slug}.jpg"
        out = ROOT / rel
        if not _save_catalog_jpeg(img, out):
            kept -= 1
            continue
        rows.append(
            {
                "id": f"DF{kept:03d}",
                "title": title[:80],
                "category": category,
                "color": color_from_row,
                "image_path": rel,
                "shop_url": f"https://www.google.com/search?tbm=shop&q={quote_plus(title + ' buy')}",
            }
        )
    return _write_catalog_rows(rows)


def build_catalog_from_deepfashion(n: int = 90) -> Path:
    """Pull a balanced upper/lower/dress subset from high-res fashion product photos."""
    target = max(6, n)
    per_cat = max(2, target // 3)
    collected: List[Tuple[str, str, Image.Image, str]] = []
    last_err: Exception | None = None
    try:
        collected = _collect_from_viewer(per_cat)
    except Exception as exc:
        last_err = exc
        print(f"High-res viewer failed ({exc}). Trying streaming datasets…")

    if len(collected) < per_cat * 2:
        extra, stream_err = _collect_from_streaming(per_cat)
        last_err = stream_err or last_err
        have = {(t, c) for t, c, _, _ in collected}
        for item in extra:
            if (item[0], item[1]) not in have:
                collected.append(item)

    if not collected:
        print(
            f"High-res catalog download failed ({last_err}). Falling back to placeholder catalog."
        )
        return build_catalog(min(n, len(CATALOG_SPEC)))

    return _write_collected_items(collected)


def main():
    parser = argparse.ArgumentParser(description="Build virtual try-on catalog")
    parser.add_argument("--n", type=int, default=90, help="Number of catalog items")
    parser.add_argument(
        "--from-deepfashion",
        action="store_true",
        help="Download high-res product photos (1080px+) instead of placeholders",
    )
    parser.add_argument(
        "--placeholders",
        action="store_true",
        help="Force generated placeholder garments (offline backup)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Only generate images/CSV (skip CLIP download)",
    )
    args = parser.parse_args()
    if args.placeholders:
        build_catalog(min(args.n, len(CATALOG_SPEC)))
    elif args.from_deepfashion or args.n > len(CATALOG_SPEC):
        build_catalog_from_deepfashion(args.n)
    else:
        build_catalog(args.n)
    if not args.skip_embeddings:
        build_embeddings()


if __name__ == "__main__":
    main()
