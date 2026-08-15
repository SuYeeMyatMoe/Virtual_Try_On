"""
VESTURE — Virtual Clothing Try-On
SegFormer + IDM-VTON (CatVTON / SD2 fallback) + FashionCLIP + Gemini

Design system: Stitch "VESTURE Digital Atelier"
Display: Bodoni Moda · Body: Hanken Grotesk · Tech: JetBrains Mono
Palette: ink black, surface charcoal, electric violet, success green
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from src.confidence import DEFAULT_GATE, evaluate_segmentation_gate, summarize_scores
from src.hf_auth import ensure_hf_login
from src.llm_advisor import caption_garment, explain_result, style_advice
from src.preprocess import load_rgb, preprocess_garment, preprocess_person, quality_check, normalize_garment_region
from src.recommend import clip_similarity, crop_by_mask, recommend_top_k
from src.segmentation import colorize_labels, infer_garment_category, segment_clothing
from src.stylist import (
    analyze_avatar,
    avatar_label_map,
    build_analysis_boards,
    catalog_for_avatar,
    catalog_for_query,
    compact_body_copy,
    interpret_chat,
    reply_to_shopper,
    resolve_presentation,
    swatch_hex,
)
from src.tryon import list_demo_pairs, try_on_vton

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=True)
ensure_hf_login()
CATALOG_CSV = ROOT / "data" / "catalog.csv"
WISHLIST_PATH = ROOT / "data" / "wishlist.json"
LOGO_PATH = ROOT / "assets" / "logo.png"

# Stitch-style editorial stills (user lookbook references)
EDITORIAL_LOOKS = [
    {
        "file": "screen1 - Copy.png",
        "id": "VE003",
        "eyebrow": "Digital exclusive",
        "title": "Neon couture drape",
        "blurb": "Woven silk with circuit light",
        "category": "dress",
        "color": "violet",
    },
    {
        "file": "screen5.png",
        "id": "VE004",
        "eyebrow": "Still life",
        "title": "Noir leather messenger",
        "blurb": "Quiet luxury still life.",
        "category": "upper",
        "color": "black",
    },
    {
        "file": "screen4.png",
        "id": "VE005",
        "eyebrow": "Studio form",
        "title": "Atelier black blazer",
        "blurb": "Catalog-standard tailoring.",
        "category": "upper",
        "color": "black",
    },
]


def _look_path(filename: str) -> Path | None:
    for folder in (ROOT / "data" / "catalog" / "images", ROOT / "assets" / "home"):
        path = folder / filename
        if path.exists():
            return path
    return None


WEB_DIR = ROOT / "assets" / "home" / "web"
THUMB_DIR = ROOT / "assets" / ".thumbs"


def _source_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _look_preview_path(filename: str) -> Path | None:
    """Small JPEG for UI; originals stay for Studio try-on."""
    original = _look_path(filename)
    web = WEB_DIR / (Path(filename).stem + ".jpg")
    if web.exists() and (original is None or _source_mtime(web) >= _source_mtime(original)):
        return web
    return original


@st.cache_data(max_entries=80, show_spinner=False)
def _jpeg_preview(src: str, _mtime: float, max_width: int = 720, quality: int = 70) -> bytes:
    """Resize and JPEG-compress so Streamlit does not ship full PNG/WebP on every rerun."""
    img = Image.open(src).convert("RGB")
    w, h = img.size
    if w > max_width:
        h = int(h * max_width / w)
        img = img.resize((max_width, h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _ensure_display_image(path: Path, max_width: int = 720) -> Path:
    """Prefer a small on-disk JPEG so the browser never downloads 1MB+ PNGs."""
    path = Path(path)
    web = WEB_DIR / f"{path.stem}.jpg"
    if web.exists() and _source_mtime(web) >= _source_mtime(path):
        return web
    try:
        size = path.stat().st_size
    except OSError:
        return path
    if path.suffix.lower() in {".jpg", ".jpeg"} and size < 180_000:
        return path
    if size < 150_000:
        return path
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    dest = THUMB_DIR / f"{path.stem}_{max_width}.jpg"
    if dest.exists() and dest.stat().st_mtime >= path.stat().st_mtime:
        return dest
    dest.write_bytes(_jpeg_preview(str(path), _source_mtime(path), max_width=max_width))
    return dest


@st.cache_data(max_entries=80, show_spinner=False)
def _fixed_aspect_jpeg(
    src: str, max_width: int = 720, ratio: tuple[int, int] = (3, 4), quality: int = 86, _mtime: float = 0.0, _v: int = 2
) -> bytes:
    """Center-crop to a shared aspect so catalog tiles share one size."""
    img = Image.open(src).convert("RGB")
    tw, th = ratio
    w, h = img.size
    target = tw / th
    current = w / h
    if current > target:
        new_w = max(1, int(h * target))
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif current < target:
        new_h = max(1, int(w / target))
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    target_w = max(1, int(max_width))
    target_h = max(1, int(target_w * th / tw))
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def _nav_logo_uri() -> str | None:
    path = LOGO_PATH
    if not path.exists():
        path = ROOT / "data" / "catalog" / "images" / "vesture_primary_logo.png"
    if not path.exists():
        return None
    img = Image.open(path)
    img.thumbnail((80, 80), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@st.cache_data(max_entries=80, show_spinner=False)
def _square_thumb(src: str, size: int = 240, quality: int = 88, _mtime: float = 0.0) -> bytes:
    """Sharp square thumbnail (2x pixels for a 120px tile)."""
    img = Image.open(src).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _show_photo(
    path: Path, *, max_width: int = 720, width="stretch", aspect: tuple[int, int] | None = None
) -> None:
    path = Path(path)
    if aspect is not None:
        st.image(
            _fixed_aspect_jpeg(
                str(path), max_width=max_width, ratio=aspect, _mtime=_source_mtime(path)
            ),
            width=width,
        )
        return
    st.image(str(_ensure_display_image(path, max_width)), width=width)


def _array_jpeg(img, max_width: int = 720, quality: int = 72) -> bytes:
    """Streamlit encodes numpy arrays as PNG; JPEG is much smaller and faster."""
    pil = img if isinstance(img, Image.Image) else Image.fromarray(img)
    if pil.mode not in {"RGB", "L"}:
        pil = pil.convert("RGB")
    w, h = pil.size
    if w > max_width:
        h = int(h * max_width / w)
        pil = pil.resize((max_width, h), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def _show_array(img, *, max_width: int = 720, caption: str | None = None, width="stretch", slot=None) -> None:
    target = slot if slot is not None else st
    target.image(_array_jpeg(img, max_width=max_width), caption=caption, width=width)


@st.cache_data(show_spinner=False)
def _load_catalog(_mtime: float):
    import pandas as pd

    return pd.read_csv(CATALOG_CSV)


st.set_page_config(
    page_title="VESTURE — Virtual Try-On Studio",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🧵",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ────────────────────────────────────────────────────────────────────────────
# Design system — fonts, palette, component CSS
# ────────────────────────────────────────────────────────────────────────────
def _inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda:ital,opsz,wght@0,6..96,500;0,6..96,600;0,6..96,700;1,6..96,500&family=Hanken+Grotesk:ital,wght@0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root{
            --paper:#131313; --paper-deep:#0E0E0E; --card:#20201F;
            --ink:#E5E2E1; --ink-soft:#CBC3D7; --ink-faint:#958EA0;
            --rust:#8B5CF6; --rust-deep:#6D3BD7; --rust-wash:rgba(139,92,246,.14);
            --sage:#10B981; --sage-deep:#059669; --sage-wash:rgba(16,185,129,.14);
            --violet:#8B5CF6; --violet-soft:#D0BCFF; --ink-black:#0A0A0A;
            --line:rgba(255,255,255,.10); --line-soft:rgba(255,255,255,.06);
            --shadow:0 18px 40px -22px rgba(0,0,0,.55);
            --shadow-sm:0 8px 20px -14px rgba(0,0,0,.45);
            --radius:8px;
        }

        html, body, [class^="css"], .stApp{
            font-family:'Hanken Grotesk', sans-serif;
            color:var(--ink);
        }

        .stApp{
            background-color:var(--paper);
            background-image:
                radial-gradient(circle at 12% -8%, var(--rust-wash), transparent 42%),
                radial-gradient(circle at 92% 8%, rgba(16,185,129,.08), transparent 38%);
        }

        header[data-testid="stHeader"]{
            background:transparent !important;
            border:none !important;
        }
        header[data-testid="stHeader"] [data-testid="stToolbar"],
        header[data-testid="stHeader"] [data-testid="stDecoration"],
        div[data-testid="stDecoration"],
        div[data-testid="stToolbar"],
        div[data-testid="stStatusWidget"],
        div[data-testid="stElementToolbar"],
        [data-testid="stElementToolbarButton"],
        [data-testid="StyledFullScreenButton"],
        button[title="View fullscreen"],
        .stAppToolbar{ display:none !important; }
        [data-testid="stToastContainer"]{
            display:flex !important;
            visibility:visible !important;
            z-index:999999 !important;
        }

        .block-container{
            padding-top:1.4rem;
            padding-bottom:4rem;
            padding-left:2.5rem;
            padding-right:2.5rem;
            max-width:1180px;
            animation: atelier-rise .55s cubic-bezier(.2,.7,.2,1);
        }
        /* Keep a floating chat bar on the same 1180px content column */
        [data-testid="stBottom"]{
            background:transparent !important;
            justify-content:center !important;
        }
        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"],
        [data-testid="stBottomBlockContainer"].block-container{
            max-width:1180px !important;
            width:min(1180px, 100%) !important;
            margin-left:auto !important;
            margin-right:auto !important;
            padding-left:2.5rem !important;
            padding-right:2.5rem !important;
            padding-bottom:1.1rem !important;
        }
        [data-testid="stChatInput"]{
            max-width:100% !important;
        }
        @keyframes atelier-rise{
            from{ opacity:0; transform:translateY(10px); }
            to{ opacity:1; transform:translateY(0); }
        }

        h1,h2,h3,h4{ font-family:'Bodoni Moda', serif; color:var(--ink); letter-spacing:-.02em; }
        code, pre, .mono{ font-family:'JetBrains Mono', monospace !important; }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"]{
            background:linear-gradient(180deg, #0A0A0A 0%, #131313 100%);
            border-right:1px solid rgba(255,255,255,.08);
        }
        section[data-testid="stSidebar"] *{ color:#E5E2E1 !important; }
        section[data-testid="stSidebar"] .block-container{ padding-top:2rem; animation:none; }

        .brand-mark{
            font-family:'Bodoni Moda', serif; font-weight:700;
            font-size:2rem; line-height:1.05; color:#E5E2E1 !important; margin-bottom:.15rem;
            letter-spacing:-.04em;
        }
        .brand-eyebrow{
            font-family:'JetBrains Mono', monospace; font-size:.68rem;
            letter-spacing:.22em; text-transform:uppercase; color:#8B5CF6 !important;
            margin-bottom:.4rem; display:block;
        }
        .brand-tagline{
            font-size:.78rem; color:#958EA0 !important; line-height:1.5;
            border-top:1px solid rgba(255,255,255,.10); margin-top:.9rem; padding-top:.9rem;
        }

        /* Hide empty sidebar; nav lives in the custom top bar */
        section[data-testid="stSidebar"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="collapsedControl"]{
            display:none !important;
        }
        .vesture-nav{
            display:flex; align-items:center; justify-content:space-between;
            gap:1rem; padding:.35rem 0 1rem 0;
            border-bottom:1px solid rgba(255,255,255,.08); margin-bottom:1.4rem;
        }
        div[data-testid="stPageLink"] a{
            font-family:'Hanken Grotesk', sans-serif !important;
            text-transform:uppercase !important;
            letter-spacing:.12em !important;
            font-size:.72rem !important;
            font-weight:700 !important;
            color:#CBC3D7 !important;
            text-decoration:none !important;
            border-bottom:2px solid transparent !important;
            padding:.35rem .1rem !important;
        }
        div[data-testid="stPageLink"] a:hover{
            color:#E5E2E1 !important;
        }
        div[data-testid="stPageLink"] a[aria-current="page"],
        div[data-testid="stPageLink"][aria-current="page"] a,
        div[data-testid="stPageLink"] a[href][aria-current="true"]{
            color:#8B5CF6 !important;
            border-bottom-color:#8B5CF6 !important;
        }

        /* Radio-as-nav (unused when top nav is on, kept as fallback) */
        section[data-testid="stSidebar"] div[role="radiogroup"]{ gap:.25rem; margin-top:1.6rem; }
        section[data-testid="stSidebar"] div[role="radiogroup"] label{
            background:transparent; border-radius:4px; padding:.55rem .75rem !important;
            border:1px solid transparent; transition:.18s ease;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
            background:rgba(255,255,255,.05); border-color:rgba(255,255,255,.10);
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
            background:rgba(139,92,246,.18); border-color:var(--violet);
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label p{
            font-family:'Hanken Grotesk', sans-serif; font-weight:700; font-size:.72rem;
            text-transform:uppercase; letter-spacing:.12em;
        }

        /* ---------- Section headers ---------- */
        .eyebrow{
            font-family:'JetBrains Mono', monospace; font-size:.72rem; letter-spacing:.24em;
            text-transform:uppercase; color:var(--violet); display:inline-block; margin-bottom:.55rem;
        }
        .section-title{ font-size:2rem; margin:0 0 .3rem 0; }
        .section-sub{ color:var(--ink-soft); font-size:1rem; max-width:640px; margin:0 0 1.4rem 0; }
        .hr-rule{ border:none; border-top:1px solid var(--line); margin:2.2rem 0; }

        /* ---------- Cards ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]){
            background:var(--card); border:1px solid var(--line) !important;
            border-radius:var(--radius) !important; box-shadow:var(--shadow-sm);
            padding:.2rem .1rem;
        }

        .step-card{
            background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.10); border-radius:var(--radius);
            padding:1.4rem 1.3rem; height:100%; backdrop-filter:blur(20px);
            transition:.2s ease; position:relative; overflow:hidden;
        }
        .step-card:before{
            content:''; position:absolute; top:0; left:0; right:0; height:1px;
            background:linear-gradient(90deg, transparent, var(--violet));
            opacity:.55;
        }
        .step-card:hover{ transform:translateY(-3px); border-color:rgba(139,92,246,.45); }
        .step-num{
            font-family:'JetBrains Mono', monospace; font-size:.72rem; color:var(--violet);
            display:block; margin-bottom:.55rem; letter-spacing:.08em;
        }
        .step-card h4{ margin:.1rem 0 .4rem 0; font-size:1.35rem; font-family:'Bodoni Moda', serif; }
        .step-card p{ color:var(--ink-soft); font-size:.9rem; line-height:1.55; margin:0; }
        .hero-steps{ margin:.35rem 0 0 0; }

        .pipe-row{
            display:flex; align-items:flex-start; gap:.85rem; padding:.85rem 0;
            border-bottom:1px solid var(--line-soft);
        }
        .pipe-row:last-child{ border-bottom:none; }
        .pipe-idx{
            font-family:'JetBrains Mono', monospace; color:#fff; background:var(--violet);
            width:26px; height:26px; min-width:26px; border-radius:50%; display:flex;
            align-items:center; justify-content:center; font-size:.75rem; margin-top:.1rem;
        }
        .pipe-text{ font-family:'JetBrains Mono', monospace; font-size:.86rem; color:var(--ink-soft); line-height:1.6; }
        .pipe-text b{ color:var(--ink); }

        /* ---------- Buttons ---------- */
        .stButton > button, .stLinkButton > a, a[data-testid="stLinkButtonLink" i]{
            border-radius:4px !important; font-weight:700 !important; font-size:.72rem !important;
            text-transform:uppercase; letter-spacing:.12em; padding:.7rem 1.4rem !important;
            transition:.18s ease !important; border:1px solid #E5E2E1 !important;
            white-space:nowrap !important;
        }
        div[data-testid="stButton"] > button[kind="primary"], button[kind="primary"]{
            background:#0A0A0A !important; border-color:#E5E2E1 !important; color:#fff !important;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover, button[kind="primary"]:hover{
            background:var(--violet) !important; border-color:var(--violet) !important;
            transform:translateY(-1px);
        }
        div[data-testid="stButton"] > button[kind="secondary"], button[kind="secondary"]{
            background:transparent !important; color:var(--ink) !important;
        }
        div[data-testid="stButton"] > button[kind="secondary"]:hover, button[kind="secondary"]:hover{
            background:var(--violet) !important; border-color:var(--violet) !important; color:#fff !important;
        }
        .stLinkButton > a{ background:transparent !important; color:var(--ink) !important; width:100%; text-align:center; }
        .stLinkButton > a:hover{ background:var(--violet) !important; border-color:var(--violet) !important; color:#fff !important; }

        /* ---------- File uploader ---------- */
        [data-testid="stFileUploader"] section, [data-testid="stFileUploaderDropzone"]{
            background:var(--card) !important; border:1px dashed #494454 !important;
            border-radius:4px !important;
        }
        [data-testid="stFileUploader"] section:hover, [data-testid="stFileUploaderDropzone"]:hover{
            border-color:var(--violet) !important; background:rgba(255,255,255,.02) !important;
        }

        /* ---------- Inputs ---------- */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div{
            background:var(--card) !important; border-radius:9px !important;
            border:1px solid var(--line) !important; color:var(--ink) !important;
        }
        .stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:focus-within > div{
            border-color:var(--violet) !important; box-shadow:0 0 0 1px var(--violet) !important;
        }
        .stCheckbox label p{ font-size:.88rem; color:var(--ink-soft); }
        .stCheckbox svg{ color:var(--violet) !important; }

        /* ---------- Alerts ---------- */
        div[data-testid="stAlertContainer"], div[data-baseweb="notification"]{
            border-radius:var(--radius) !important; border:1px solid var(--line) !important;
        }

        /* ---------- Images ---------- */
        div[data-testid="stImage"] img{
            border-radius:12px; border:1px solid var(--line); box-shadow:var(--shadow-sm);
        }
        .catalog-tile div[data-testid="stImage"],
        .catalog-tile div[data-testid="stImage"] > img,
        .catalog-tile img{
            width:100% !important; aspect-ratio:3 / 4 !important;
            object-fit:cover !important; height:auto !important;
        }
        .wishlist-tile div[data-testid="stImage"],
        .wishlist-tile div[data-testid="stImage"] > img,
        .wishlist-tile img{
            width:120px !important;
            height:120px !important;
            max-width:120px !important;
            max-height:120px !important;
            aspect-ratio:1 / 1 !important;
            object-fit:cover !important;
        }
        .wishlist-tile p, .wishlist-tile [data-testid="stMarkdownContainer"] p{
            font-size:.78rem !important;
            line-height:1.25 !important;
            margin-bottom:.15rem !important;
        }
        div[data-testid="stImageCaption"], div[data-testid="stImage"] figcaption{
            font-family:'JetBrains Mono', monospace; font-size:.72rem !important;
            letter-spacing:.06em; text-transform:uppercase; color:var(--ink-faint) !important;
            text-align:center; margin-top:.4rem;
        }

        /* ---------- Score bars ---------- */
        .score-row{ margin-bottom:1rem; }
        .score-row-top{ display:flex; justify-content:space-between; margin-bottom:.3rem; }
        .score-label{ font-size:.86rem; font-weight:600; color:var(--ink); }
        .score-value{ font-family:'JetBrains Mono', monospace; font-size:.86rem; color:var(--ink-soft); }
        .score-track{ height:2px; background:var(--paper-deep); border-radius:0; overflow:hidden; }
        .score-fill{ height:100%; border-radius:99px; transition:width .5s ease; }

        /* ---------- Pills ---------- */
        .pill{
            display:inline-block; font-family:'JetBrains Mono', monospace; font-size:.72rem;
            letter-spacing:.05em; text-transform:uppercase; padding:.25rem .65rem;
            border-radius:4px; font-weight:600;
        }
        .pill-ok{ background:var(--sage-wash); color:var(--sage-deep); border:1px solid var(--sage); }
        .pill-bad{ background:var(--rust-wash); color:var(--rust-deep); border:1px solid var(--rust); }
        .pill-neutral{ background:var(--paper-deep); color:var(--ink-soft); border:1px solid var(--line); }
        .palette-row{ display:flex; flex-wrap:wrap; gap:.45rem; margin:.35rem 0 .85rem; }
        .swatch{
            display:inline-flex; align-items:center; gap:.4rem;
            font-family:'JetBrains Mono', monospace; font-size:.7rem;
            letter-spacing:.04em; text-transform:uppercase;
            padding:.28rem .6rem; border-radius:4px;
            border:1px solid var(--line); background:rgba(255,255,255,.04); color:var(--ink-soft);
        }
        .swatch i{
            width:12px; height:12px; border-radius:50%; display:inline-block;
            border:1px solid rgba(255,255,255,.28);
        }
        .color-set{ display:flex; flex-wrap:wrap; gap:.85rem; margin:.35rem 0 .4rem; }
        .color-chip{ display:flex; flex-direction:column; align-items:center; gap:.45rem; min-width:88px; }
        .color-chip-swatch{
            width:76px; height:76px; border-radius:8px;
            border:1px solid rgba(255,255,255,.18);
            box-shadow:var(--shadow-sm);
        }
        .color-chip-label{
            font-family:'JetBrains Mono', monospace; font-size:.68rem;
            letter-spacing:.06em; text-transform:uppercase; color:var(--ink-soft);
            text-align:center; max-width:96px;
        }
        .analysis-kicker{
            font-family:'JetBrains Mono', monospace; font-size:.62rem; letter-spacing:.08em;
            text-transform:uppercase; color:var(--violet); margin:0 0 .35rem 0;
        }
        .analysis-card div[data-testid="stImage"] img{
            border-radius:6px;
            border:1px solid var(--line);
        }

        /* ---------- Product cards ---------- */
        .product-card{
            background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
            padding:.8rem; box-shadow:var(--shadow-sm); transition:.2s ease; height:100%;
        }
        .product-card:hover{ transform:translateY(-4px); box-shadow:var(--shadow); }
        .product-title{
            font-weight:700; font-size:.88rem; margin:.5rem 0 .2rem 0;
            min-height:2.6em; line-height:1.3;
            display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
            overflow:hidden;
        }
        .product-meta{ font-size:.76rem; color:var(--ink-faint); margin-bottom:.5rem; min-height:1.2em; }

        /* ---------- Before / after sample cards ---------- */
        .ba-card{
            background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
            padding:.85rem .85rem 1rem; box-shadow:var(--shadow-sm); height:100%;
        }
        .ba-label{
            font-family:'JetBrains Mono', monospace; font-size:.68rem; letter-spacing:.08em;
            text-transform:uppercase; color:var(--violet); display:inline-block; margin-bottom:.35rem;
        }
        .ba-card h4{ font-family:'Bodoni Moda', serif; font-size:1.05rem; margin:.55rem 0 .2rem 0; }
        .ba-card p{ margin:0; color:var(--ink-soft); font-size:.86rem; }

        /* ---------- Footer ---------- */
        .atelier-footer{
            margin-top:3rem; padding-top:1.4rem; border-top:1px solid var(--line);
            font-family:'JetBrains Mono', monospace; font-size:.72rem; letter-spacing:.08em;
            text-transform:uppercase; color:var(--ink-faint); text-align:center;
        }

        div[data-testid="stMetricValue"]{ font-family:'Bodoni Moda', serif; }

        .sys-ready{
            font-family:'JetBrains Mono', monospace; font-size:.72rem; letter-spacing:.08em;
            color:var(--violet); display:inline-flex; align-items:center; gap:.45rem;
        }
        .sys-dot{
            width:8px; height:8px; border-radius:50%; background:var(--violet);
            box-shadow:0 0 0 4px rgba(139,92,246,.25);
        }
        .look-cap{
            font-family:'JetBrains Mono', monospace; font-size:.62rem; letter-spacing:.08em;
            text-transform:uppercase; color:var(--violet); margin:.45rem 0 .1rem 0;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
        }
        .look-title{
            font-family:'Bodoni Moda', serif; font-size:.95rem; margin:0 0 .12rem 0;
            line-height:1.25; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
        }
        .look-blurb{
            color:var(--ink-soft); font-size:.78rem; margin:0 0 .35rem 0; line-height:1.3;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
        }
        .nav-brand-fallback{
            font-family:'Bodoni Moda', serif; font-weight:700; letter-spacing:-.04em;
        }
        .nav-brand{
            display:flex; align-items:center; gap:.65rem;
        }
        .nav-brand img{
            height:40px; width:40px; object-fit:contain; display:block;
            border:none !important; box-shadow:none !important; border-radius:6px;
        }
        .st-key-home_enter_studio{
            position:absolute !important;
            left:12%; right:12%; bottom:1.35rem;
            width:auto !important; z-index:3;
        }
        [data-testid="stColumn"]:has(.st-key-home_enter_studio){
            position:relative !important;
        }
        .st-key-home_enter_studio [data-testid="stButton"] > button{
            width:100% !important;
            box-shadow:0 14px 32px -12px rgba(0,0,0,.75);
        }
        /* Compact Try / Save / Buy on one catalog row */
        div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"]
        div[data-testid="stButton"] > button,
        div[data-testid="stHorizontalBlock"] div[data-testid="stHorizontalBlock"]
        a[data-testid="stLinkButtonLink" i]{
            padding:.55rem .3rem !important;
            font-size:.6rem !important;
            letter-spacing:.06em !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ────────────────────────────────────────────────────────────────────────────
# Small render helpers
# ────────────────────────────────────────────────────────────────────────────
def _section_head(eyebrow: str, title: str, subtitle: str = "") -> None:
    sub_html = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <span class="eyebrow">{eyebrow}</span>
        <h2 class="section-title">{title}</h2>
        {sub_html}
        """,
        unsafe_allow_html=True,
    )


def _pill(text: str, tone: str = "neutral") -> str:
    return f'<span class="pill pill-{tone}">{text}</span>'


def _score_bar(label: str, value: float, threshold: float | None = None) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    tone = "var(--violet)"
    if threshold is not None:
        tone = "var(--sage)" if value >= threshold else "var(--violet)"
    return f"""
    <div class="score-row">
      <div class="score-row-top">
        <span class="score-label">{label}</span>
        <span class="score-value">{pct:.1f}%</span>
      </div>
      <div class="score-track"><div class="score-fill" style="width:{pct:.1f}%; background:{tone};"></div></div>
    </div>
    """


# ────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SegFormer clothing model…")
def _warm_segformer():
    from src.segmentation import load_segformer

    return load_segformer()


def _render_llm_panels(caption: str, advice: str, explanation: str, *, used_gemini: bool = False) -> None:
    st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)
    _section_head("Stylist layer", "AI description &amp; advice")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("**AI garment description**")
            st.write(caption or "—")
    with c2:
        with st.container(border=True):
            st.markdown("**Stylist advice**")
            st.write(advice or "—")
    with st.container(border=True):
        st.markdown("**Why this result**")
        st.write(explanation or "—")


def _render_reco_cards(results: list) -> None:
    if not results:
        st.warning("No results.")
        return
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(min(5, len(results)))
    for col, item in zip(cols, results):
        with col:
            img_path = ROOT / item["image_path"]
            st.markdown('<div class="product-card catalog-tile">', unsafe_allow_html=True)
            if img_path.exists():
                _show_photo(img_path, max_width=560, aspect=(3, 4))
            sim = item["similarity"]
            tone = "ok" if item["high_confidence"] else "neutral"
            st.markdown(
                f"""
                <div class="product-title">{item['title']}</div>
                <div class="product-meta">{item['category']} · {item['color']}</div>
                {_pill(f"{sim:.1%} match", tone)}
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
            st.link_button("Buy / Find online", item["shop_url"], width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)


def _render_similar_items(query_img: Image.Image, results=None):
    """Top-5 FashionCLIP recommendations under Try-On."""
    st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)
    _section_head(
        "Curated for you",
        "Similar Items",
        "Powered by Marqo FashionCLIP (fallback: OpenAI CLIP ViT-B/32).",
    )

    if not CATALOG_CSV.exists():
        st.warning("Catalog missing. Run: `python -m src.catalog_builder --from-deepfashion`")
        if st.button("Build starter catalog now", type="secondary"):
            with st.spinner("Building catalog images…"):
                from src.catalog_builder import build_catalog

                build_catalog(30)
            st.success("Catalog created. Click Recommend Top-5 again.")
        return

    qc1, qc2 = st.columns([1, 3])
    with qc1:
        st.image(_array_jpeg(query_img, max_width=440), caption="Query garment", width=220)
    with qc2:
        st.markdown(
            '<p style="color:var(--ink-soft); margin-top:1.2rem;">'
            "We'll match this garment against the catalog on visual style, "
            "silhouette and color to surface the five closest pieces.</p>",
            unsafe_allow_html=True,
        )
        recommend_clicked = st.button("Recommend Top-5", type="primary")

    if results is None:
        results = st.session_state.get("reco_results")
    if recommend_clicked:
        with st.spinner("Searching catalog…"):
            try:
                results = recommend_top_k(query_img, k=5)
                st.session_state["reco_results"] = results
            except Exception as exc:
                st.error(f"Recommendation failed: {exc}")
                return
    if results:
        _render_reco_cards(results)


def _go_studio() -> None:
    studio = st.session_state.get("page_studio")
    if studio is not None:
        st.switch_page(studio)


def _image_with_studio_cta(path: Path, *, key: str, max_width: int = 720) -> None:
    _show_photo(path, max_width=max_width)
    if st.button("Enter the studio →", type="primary", key=key, width="stretch"):
        _go_studio()


def _show_look(
    look: dict, *, key_prefix: str, max_width: int = 880, aspect: tuple[int, int] | None = None
) -> None:
    preview = _look_preview_path(look["file"])
    if preview is None:
        st.caption(f"Missing {look['file']}")
        return
    _show_photo(preview, max_width=max_width, aspect=aspect)
    st.html(
        f"""
        <div class="look-cap">{look['eyebrow']}</div>
        <div class="look-title">{look['title']}</div>
        <p class="look-blurb">{look['blurb']}</p>
        """
    )
    original = _look_path(look["file"]) or preview
    st.button(
        "Try in Studio",
        key=f"{key_prefix}_{look['id']}",
        width="stretch",
        on_click=_go_studio_with_garment,
        args=(str(original), look["title"], look.get("category")),
    )


@st.fragment
def _home_lookbook() -> None:
    cols = st.columns(len(EDITORIAL_LOOKS), gap="medium")
    for col, look in zip(cols, EDITORIAL_LOOKS):
        with col:
            _show_look(look, key_prefix="home_b", max_width=640, aspect=(3, 4))


def _pipeline_cards() -> None:
    steps = [
        ("01 // SEGMENT", "Semantic Masking", "SegFormer maps body contours and garments with pixel-perfect accuracy."),
        ("02 // TRY-ON", "Garment-conditioned fit", "IDM-VTON dresses you from the garment image (CatVTON / SD2 fallback)."),
        ("03 // RECOMMEND", "FashionCLIP + Gemini", "Visual search returns Top-5 pieces; Gemini explains the look."),
    ]
    cols = st.columns(3, gap="medium")
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.html(
                f"""
                <div class="step-card">
                    <span class="step-num">{num}</span>
                    <h4>{title}</h4>
                    <p>{desc}</p>
                </div>
                """
            )


# ────────────────────────────────────────────────────────────────────────────
# Home tab
# ────────────────────────────────────────────────────────────────────────────
def home_tab():
    hero_l, hero_r = st.columns([1.15, 1], gap="large", vertical_alignment="top")
    with hero_l:
        st.markdown(
            """
            <span class="eyebrow">Digital Atelier</span>
            <h1 style="font-size:clamp(2.4rem, 5vw, 4.4rem); line-height:1.08; margin:0 0 .7rem 0; letter-spacing:-.03em;">
                YOUR DIGITAL ATELIER.
            </h1>
            <p class="section-sub" style="font-size:1.12rem; max-width:640px;">
                The future of fashion is virtual. Try on, style, and shop with AI precision.
            </p>
            """,
            unsafe_allow_html=True,
        )
    with hero_r:
        hero_img = _look_preview_path("screen.png") or _look_path("screen.png")
        if hero_img:
            _image_with_studio_cta(hero_img, key="home_enter_studio", max_width=720)

    st.markdown('<div class="hero-steps"></div>', unsafe_allow_html=True)
    _pipeline_cards()

    st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            """
            <span class="eyebrow">The Problem</span>
            <h3 style="margin-top:0;">Fit is a guessing game</h3>
            <p style="color:var(--ink-soft); line-height:1.65;">
                Online shoppers cannot physically try clothes before buying, which leads to
                poor fit and style decisions — and high return rates.
            </p>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <span class="eyebrow">Who It's For</span>
            <h3 style="margin-top:0;">Built for the whole wardrobe</h3>
            <p style="color:var(--ink-soft); line-height:1.65;">
                Fashion e-commerce shoppers, stylists, and students exploring
                computer-vision applications.
            </p>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)
    _section_head(
        "Lookbook",
        "Stitch reference edit",
        "Editorial stills for the Digital Atelier — neon drape, still life, tailoring.",
    )
    _home_lookbook()

# ────────────────────────────────────────────────────────────────────────────
# Try-On tab
# ────────────────────────────────────────────────────────────────────────────
def tryon_tab():
    c1, c2 = st.columns([1, 1.15], gap="large", vertical_alignment="top")
    with c1:
        _section_head(
            "VTO_ENGINE_V2",
            "Studio",
            "Upload a subject photo and a garment to begin the AI styling process.",
        )
        with st.container(border=True):
            st.markdown("**Subject data**")
            st.caption("Person photo · PNG, JPG · full body, front-facing.")
            person_file = st.file_uploader(
                "Person photo (full body)",
                type=["jpg", "jpeg", "png", "webp"],
                key="person",
                label_visibility="collapsed",
            )
            stylist_avatar = st.session_state.get("stylist_avatar")
            if stylist_avatar is not None and person_file is None:
                st.caption("Using your Stylist AI avatar as the subject photo.")
                st.image(_array_jpeg(stylist_avatar, max_width=280), width=180)
        with st.container(border=True):
            st.markdown("**Target garment**")
            st.caption("Garment image · flat lay preferred.")
            garment_file = st.file_uploader(
                "Garment image",
                type=["jpg", "jpeg", "png", "webp"],
                key="garment",
                label_visibility="collapsed",
            )
            catalog_pick = st.session_state.get("catalog_garment_path")
            if catalog_pick and Path(catalog_pick).exists():
                cat_label = st.session_state.get("catalog_garment_category") or ""
                title = st.session_state.get("catalog_garment_title", Path(catalog_pick).name)
                suffix = f" · {cat_label}" if cat_label else ""
                st.caption(f"Catalog pick · {title}{suffix}")
                _show_photo(Path(catalog_pick), max_width=280)
                if st.button("Clear catalog pick", type="secondary"):
                    st.session_state.pop("catalog_garment_path", None)
                    st.session_state.pop("catalog_garment_title", None)
                    st.session_state.pop("catalog_garment_category", None)
                    st.rerun()
        demo_pairs = list_demo_pairs()
        demo_choice = "Upload my own"
        if demo_pairs:
            with st.container(border=True):
                st.markdown("**Demo pair (VITON-HD)**")
                labels = ["Upload my own"] + [p["title"] for p in demo_pairs]
                demo_choice = st.selectbox("Use a benchmark pair", labels, label_visibility="collapsed")
                st.caption("Frontal studio shots — best for confidence ≥ 0.85.")
    with c2:
        studio_ref = _look_preview_path("screen2_crop.png") or _look_path("screen2_crop.png")
        if studio_ref:
            _show_photo(studio_ref, max_width=720)

    regions = ["upper", "lower", "dress"]
    if "studio_region" not in st.session_state:
        catalog_cat = str(st.session_state.get("catalog_garment_category") or "").strip().lower()
        if catalog_cat in regions:
            st.session_state["studio_region"] = catalog_cat
        else:
            pref = st.session_state.get("pref_region", "upper")
            st.session_state["studio_region"] = pref if pref in regions else "upper"

    with st.container(border=True):
        st.markdown("**Styling options**")
        o1, o2, o3 = st.columns(3)
        with o1:
            category = st.selectbox(
                "Garment region",
                regions,
                key="studio_region",
                help="Upper = tops. Lower = pants, jeans, shorts, skirts. Dress = full garment.",
            )
            st.caption("Lower keeps pants on the legs — not as a top.")
        with o2:
            color = st.text_input("Color (optional)", placeholder="e.g. navy blue")
        with o3:
            style = st.text_input("Style (optional)", placeholder="e.g. casual cotton")
        o4, o5 = st.columns(2)
        with o4:
            fast = st.checkbox("Fast preview size (512×384-class)", value=True)
        with o5:
            show_labels = st.checkbox("Show SegFormer label map", value=False)

    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Generate Try-On", type="primary", width="stretch")

    if run_clicked:
        demo_pair = None
        if demo_choice != "Upload my own":
            demo_pair = next((p for p in demo_pairs if p["title"] == demo_choice), None)

        if demo_pair is not None:
            person_raw = load_rgb(Image.open(demo_pair["person_path"]))
            garment = preprocess_garment(load_rgb(Image.open(demo_pair["garment_path"])))
            if demo_pair.get("category") in regions:
                category = demo_pair["category"]
        else:
            if person_file:
                person_raw = load_rgb(Image.open(person_file))
            elif st.session_state.get("stylist_avatar") is not None:
                person_raw = load_rgb(st.session_state["stylist_avatar"])
            else:
                st.error("Please upload a person photo, pick a VITON-HD demo pair, or analyze an avatar in Stylist AI.")
                return
            garment = None
            if garment_file:
                garment = preprocess_garment(load_rgb(Image.open(garment_file)))
            else:
                catalog_pick = st.session_state.get("catalog_garment_path")
                if catalog_pick and Path(catalog_pick).exists():
                    garment = preprocess_garment(load_rgb(Image.open(catalog_pick)))

        category = normalize_garment_region(category)
        raw_catalog_cat = st.session_state.get("catalog_garment_category")
        if raw_catalog_cat:
            catalog_cat = normalize_garment_region(raw_catalog_cat)
            if catalog_cat == "lower":
                category = "lower"

        ok, msg = quality_check(person_raw)
        if not ok:
            st.error(msg)
            return

        person = preprocess_person(person_raw, fast=fast)

        with st.spinner("Running SegFormer clothing segmentation…"):
            _warm_segformer()
            if garment is not None:
                try:
                    guessed = infer_garment_category(garment)
                    if guessed == "lower":
                        category = "lower"
                    elif guessed == "dress" and category != "upper":
                        category = "dress"
                except Exception:
                    pass
            mask, seg_conf, label_map = segment_clothing(person, category=category)

        ok_gate, gate_msg = evaluate_segmentation_gate(seg_conf, DEFAULT_GATE)

        st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)
        with st.container(border=True):
            g1, g2 = st.columns([3, 1])
            with g1:
                st.markdown(
                    _score_bar("Segmentation confidence", seg_conf, DEFAULT_GATE),
                    unsafe_allow_html=True,
                )
                st.caption(f"Confidence gate: {DEFAULT_GATE:.0%}")
            with g2:
                st.markdown(
                    f'<div style="text-align:right; padding-top:.4rem;">{_pill("PASS" if ok_gate else "FAIL", "ok" if ok_gate else "bad")}</div>',
                    unsafe_allow_html=True,
                )

        if ok_gate:
            st.success(gate_msg)
        else:
            st.warning(gate_msg)
            st.info("Try-on is blocked until segmentation confidence ≥ 0.85. Showing mask only.")

        v1, v2, v3 = st.columns(3)
        _show_array(person, caption="Person (preprocessed)", slot=v1)
        _show_array(mask, caption=f"Clothing mask · {category}", slot=v2)
        if show_labels:
            _show_array(colorize_labels(label_map), caption="Label map", slot=v3)
        elif garment is not None:
            _show_array(garment, caption="Garment", slot=v3)

        if not ok_gate:
            st.session_state.pop("tryon_person", None)
            st.session_state.pop("tryon_result", None)
            st.session_state.pop("query_for_reco", None)
            return

        with st.spinner("Preparing garment description…"):
            caption, cap_gemini = caption_garment(
                garment, category=category, color=color or None, style=style or None
            )

        with st.spinner(
            "Running CatVTON lower-body try-on…"
            if category == "lower"
            else "Running CatVTON dress try-on…"
            if category == "dress"
            else "Running IDM-VTON (CatVTON / SD2 fallback if the Space is busy)…"
        ):
            result, warn, prompt, engine = try_on_vton(
                person,
                mask,
                category=category,
                color=color or None,
                style=style or None,
                extra_prompt=caption,
                garment=garment,
                use_demo_fallback=True,
            )

        st.markdown(
            f'<p class="mono" style="font-size:.8rem; color:var(--ink-soft); '
            f'background:var(--paper-deep); border-left:3px solid var(--violet); '
            f'padding:.5rem .8rem; border-radius:6px;">Engine: {engine} · Prompt: {prompt}</p>',
            unsafe_allow_html=True,
        )
        if warn:
            st.warning(warn)
        if result is None:
            st.error("Try-on failed. Set HF_TOKEN in .env or add a fallback image under assets/demo/.")
            return

        with st.spinner("Computing try-on confidence (FashionCLIP)…"):
            tryon_crop = crop_by_mask(result, mask)
            ref = garment if garment is not None else crop_by_mask(person, mask)
            try:
                sim = clip_similarity(ref, tryon_crop)
            except Exception as exc:
                st.warning(f"CLIP similarity unavailable ({exc}); using 0.5 placeholder.")
                sim = 0.5
            scores = summarize_scores(seg_conf, sim, mask, DEFAULT_GATE)

        query_img = garment if garment is not None else tryon_crop
        reco_results = []
        try:
            reco_results = recommend_top_k(query_img, k=5)
        except Exception:
            reco_results = []

        with st.spinner("Writing stylist advice…"):
            advice, adv_gemini = style_advice(
                caption, category=category, top5=reco_results, scores=scores
            )
            explanation, exp_gemini = explain_result(
                scores=scores, top5=reco_results, engine=engine, garment_desc=caption
            )
        used_gemini = cap_gemini or adv_gemini or exp_gemini

        st.session_state["tryon_person"] = person
        st.session_state["tryon_result"] = result
        st.session_state["tryon_scores"] = scores
        st.session_state["query_for_reco"] = query_img
        st.session_state["reco_results"] = reco_results
        st.session_state["tryon_caption"] = caption
        st.session_state["tryon_advice"] = advice
        st.session_state["tryon_explain"] = explanation
        st.session_state["tryon_gemini"] = used_gemini
        st.session_state["tryon_engine"] = engine
        st.session_state["tryon_prompt"] = prompt

        st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)
        _section_head("AI Render", "Before &amp; After")

        r1, r2 = st.columns(2)
        with r1:
            _show_array(person, caption="Before")
        with r2:
            _show_array(result, caption=f"After ({engine})")

        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown(_score_bar("Seg confidence", scores["seg_conf"]), unsafe_allow_html=True)
                st.markdown(_score_bar("CLIP similarity", scores["clip_sim"]), unsafe_allow_html=True)
            with sc2:
                st.markdown(_score_bar("Mask quality", scores["mask_quality"]), unsafe_allow_html=True)
                st.markdown(
                    _score_bar("Try-on confidence", scores["tryon_conf"], 0.85),
                    unsafe_allow_html=True,
                )

        if scores["passes_tryon_gate"]:
            st.success(f"Try-on confidence {scores['tryon_conf']:.2%} meets the 0.85 gate.")
        else:
            st.info(
                f"Try-on confidence {scores['tryon_conf']:.2%} is below 0.85 — "
                "result shown, but treat as lower reliability."
            )

        _render_llm_panels(caption, advice, explanation, used_gemini=used_gemini)
        _render_similar_items(query_img, results=reco_results)

    elif st.session_state.get("query_for_reco") is not None:
        # Keep before/after + similar items after a previous successful try-on
        if st.session_state.get("tryon_result") is not None:
            st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)
            engine = st.session_state.get("tryon_engine", "try-on")
            _section_head("AI Render", "Last try-on")
            r1, r2 = st.columns(2)
            person_prev = st.session_state.get("tryon_person")
            if person_prev is not None:
                _show_array(person_prev, caption="Before", slot=r1)
            _show_array(st.session_state["tryon_result"], caption=f"After ({engine})", slot=r2)
            scores = st.session_state.get("tryon_scores")
            if scores:
                with st.container(border=True):
                    sc1, sc2 = st.columns(2)
                    with sc1:
                        st.markdown(_score_bar("Seg confidence", scores["seg_conf"]), unsafe_allow_html=True)
                        st.markdown(_score_bar("CLIP similarity", scores["clip_sim"]), unsafe_allow_html=True)
                    with sc2:
                        st.markdown(_score_bar("Mask quality", scores["mask_quality"]), unsafe_allow_html=True)
                        st.markdown(
                            _score_bar("Try-on confidence", scores["tryon_conf"], 0.85),
                            unsafe_allow_html=True,
                        )
            if st.session_state.get("tryon_caption"):
                _render_llm_panels(
                    st.session_state.get("tryon_caption", ""),
                    st.session_state.get("tryon_advice", ""),
                    st.session_state.get("tryon_explain", ""),
                    used_gemini=bool(st.session_state.get("tryon_gemini")),
                )
        _render_similar_items(st.session_state["query_for_reco"])


def _load_wishlist_file() -> list[str]:
    if not WISHLIST_PATH.exists():
        return []
    try:
        import json

        data = json.loads(WISHLIST_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(x) for x in data]
    except Exception:
        pass
    return []


def _write_wishlist_file(ids: list[str]) -> None:
    import json

    WISHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    WISHLIST_PATH.write_text(json.dumps(list(ids), indent=2), encoding="utf-8")


def _ensure_wishlist() -> list[str]:
    if "wishlist" not in st.session_state:
        st.session_state.wishlist = _load_wishlist_file()
    return list(st.session_state.wishlist)


def _toggle_wish(item_id: str) -> None:
    item_id = str(item_id)
    wish = _ensure_wishlist()
    if item_id in wish:
        wish.remove(item_id)
    else:
        wish.append(item_id)
    st.session_state.wishlist = wish
    try:
        _write_wishlist_file(wish)
    except Exception:
        pass


def _save_piece(item_id: str, title: str = "") -> None:
    item_id = str(item_id)
    already = item_id in _ensure_wishlist()
    if already:
        return
    _toggle_wish(item_id)
    name = title or "this piece"
    st.session_state["wish_notice"] = ("saved", f"Saved “{name}” to Catalog.")


def _remove_piece(item_id: str, title: str = "") -> None:
    item_id = str(item_id)
    wish = _ensure_wishlist()
    if item_id not in wish:
        return
    wish.remove(item_id)
    st.session_state.wishlist = wish
    try:
        _write_wishlist_file(wish)
    except Exception:
        pass
    name = title or "this piece"
    st.session_state["wish_notice"] = ("removed", f"Removed “{name}” from saved pieces.")


def _flush_wish_notice(slot=None) -> None:
    notice = st.session_state.pop("wish_notice", None)
    if not (isinstance(notice, tuple) and len(notice) == 2):
        return
    kind, msg = notice
    icon = ":material/bookmark:" if kind == "saved" else ":material/bookmark_remove:"
    st.toast(msg, icon=icon, duration="long")
    target = slot if slot is not None else st
    if kind == "saved":
        target.success(msg, icon=icon)
    else:
        target.info(msg, icon=icon)


def _render_saved_pieces(*, key_prefix: str, compact: bool = False) -> None:
    wish = _ensure_wishlist()
    if not wish:
        st.caption("Nothing saved yet. Tap Save on a catalog piece.")
        return
    if not CATALOG_CSV.exists():
        st.caption("Catalog file is missing, so saved pieces cannot be shown.")
        return
    df = _load_catalog(CATALOG_CSV.stat().st_mtime)
    saved = df[df["id"].astype(str).isin(wish)]
    if saved.empty:
        st.caption("Saved IDs were not found in the catalog. Try Save again on a shop piece.")
        return

    rows = list(saved.itertuples(index=False))
    n_cols = 6
    for start in range(0, len(rows), n_cols):
        cols = st.columns(n_cols, gap="small")
        for col, row in zip(cols, rows[start : start + n_cols]):
            img_path = ROOT / str(row.image_path)
            with col:
                st.markdown('<div class="wishlist-tile">', unsafe_allow_html=True)
                if img_path.exists():
                    st.image(
                        _square_thumb(str(img_path), size=240, _mtime=_source_mtime(img_path)),
                        width=120,
                    )
                st.markdown(f"**{row.title}**")
                st.caption(f"{row.category} · {row.color}")
                st.button(
                    "Remove",
                    key=f"{key_prefix}_{row.id}",
                    width="stretch",
                    on_click=_remove_piece,
                    args=(str(row.id), str(row.title)),
                )
                st.link_button("Buy", str(row.shop_url), width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)


def _go_studio_with_garment(
    img_path: Path | str, title: str, category: str | None = None
) -> None:
    st.session_state["catalog_garment_path"] = str(img_path)
    st.session_state["catalog_garment_title"] = title
    cat = str(category or "").strip().lower()
    if cat in {"upper", "lower", "dress"}:
        st.session_state["catalog_garment_category"] = cat
        st.session_state["studio_region"] = cat
    studio = st.session_state.get("page_studio")
    if studio is not None:
        st.switch_page(studio)


def _swatch_hex(name: str) -> str:
    return swatch_hex(name)


def _palette_html(colors: list) -> str:
    chips = []
    for color in colors:
        label = html.escape(str(color))
        chips.append(
            f'<span class="swatch"><i style="background:{_swatch_hex(str(color))}"></i>{label}</span>'
        )
    return f'<div class="palette-row">{"".join(chips)}</div>'


def _color_set_html(colors: list) -> str:
    chips = []
    for color in colors:
        label = html.escape(str(color))
        chips.append(
            '<div class="color-chip">'
            f'<span class="color-chip-swatch" style="background:{_swatch_hex(str(color))}"></span>'
            f'<span class="color-chip-label">{label}</span>'
            "</div>"
        )
    return f'<div class="color-set">{"".join(chips)}</div>'


def _init_stylist_state() -> None:
    st.session_state.setdefault("stylist_messages", [])
    st.session_state.setdefault("stylist_recs", [])
    st.session_state.setdefault("stylist_all_recs", [])
    st.session_state.setdefault("stylist_shown_ids", [])
    st.session_state.setdefault("stylist_analysis", None)
    st.session_state.setdefault("stylist_label_map", None)
    st.session_state.setdefault("stylist_color_board", None)
    st.session_state.setdefault("stylist_body_board", None)
    st.session_state.setdefault("stylist_board_presentation", None)


def _refresh_stylist_boards(avatar: Image.Image, analysis: dict, presentation: str) -> None:
    label_map = st.session_state.get("stylist_label_map")
    color_board, body_board = build_analysis_boards(
        avatar,
        analysis,
        label_map=label_map,
        presentation=presentation,
    )
    st.session_state["stylist_color_board"] = color_board
    st.session_state["stylist_body_board"] = body_board
    st.session_state["stylist_board_presentation"] = presentation


def _remember_recs(items: list) -> None:
    shown = st.session_state.setdefault("stylist_shown_ids", [])
    all_recs = st.session_state.setdefault("stylist_all_recs", [])
    known = {str(r.get("id")) for r in all_recs}
    for item in items:
        item_id = str(item.get("id"))
        if item_id not in shown:
            shown.append(item_id)
        if item_id not in known:
            all_recs.append(item)
            known.add(item_id)
    st.session_state["stylist_recs"] = items


def _render_stylist_looks(results: list, *, key_prefix: str) -> None:
    if not results:
        st.info("No catalog pieces yet. Analyze an avatar or ask the stylist for a look.")
        return
    cols = st.columns(min(5, len(results)))
    for col, item in zip(cols, results):
        with col:
            img_path = ROOT / item["image_path"]
            st.markdown('<div class="product-card catalog-tile">', unsafe_allow_html=True)
            if img_path.exists():
                _show_photo(img_path, max_width=560, aspect=(3, 4))
            tone = "ok" if item.get("high_confidence") else "neutral"
            st.markdown(
                f"""
                <div class="product-title">{html.escape(str(item['title']))}</div>
                <div class="product-meta">{html.escape(str(item['category']))} · {html.escape(str(item['color']))}</div>
                {_pill(f"{item['similarity']:.1%} match", tone)}
                """,
                unsafe_allow_html=True,
            )
            if st.button("Try in Studio", key=f"{key_prefix}_{item['id']}", width="stretch"):
                _go_studio_with_garment(img_path, str(item["title"]), str(item.get("category") or ""))
            st.link_button("Buy / Find online", item["shop_url"], width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)


def catalog_tab():
    _section_head(
        "Atelier shop",
        "Catalog",
        "Browse the VESTURE edit — try a piece in Studio, save it, or shop it online.",
    )
    notice_slot = st.empty()

    _section_head(
        "Digital exclusive",
        "Stitch reference looks",
        "Editorial photography used as the atelier design language.",
    )
    _catalog_featured()

    st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)

    if not CATALOG_CSV.exists():
        st.warning("Catalog missing. Run: `python -m src.catalog_builder`")
        if st.button("Build starter catalog now", type="secondary"):
            with st.spinner("Building catalog images…"):
                from src.catalog_builder import build_catalog

                build_catalog(30)
            st.success("Catalog created. Refresh this page.")
        return

    _catalog_grid()
    _flush_wish_notice(slot=notice_slot)


@st.fragment
def _catalog_featured() -> None:
    feat_cols = st.columns(len(EDITORIAL_LOOKS), gap="small")
    for col, look in zip(feat_cols, EDITORIAL_LOOKS):
        with col:
            _show_look(look, key_prefix="cat_feat", max_width=720, aspect=(3, 4))


def _catalog_grid() -> None:
    df = _load_catalog(CATALOG_CSV.stat().st_mtime)
    df = df[~df["id"].astype(str).str.startswith("VE")]
    df = df[~df["color"].astype(str).str.lower().eq("white")]
    df = df[~df["id"].astype(str).eq("DF030")]
    filt = st.segmented_control(
        "Category",
        options=["All", "Tops", "Bottoms", "Dresses"],
        default="All",
    )
    st.html(
        """
        <style>
        .catalog-tile div[data-testid="stImage"] img{
            width:100% !important;
            aspect-ratio:3 / 4 !important;
            object-fit:cover !important;
            height:auto !important;
        }
        </style>
        """
    )
    mapping = {"Tops": "upper", "Bottoms": "lower", "Dresses": "dress"}
    if filt in mapping:
        df = df[df["category"].astype(str) == mapping[filt]]

    st.caption(f"{len(df)} pieces in this edit")
    if df.empty:
        st.info("No pieces in this category.")
    else:
        wish = _ensure_wishlist()
        n_cols = 3
        cols = st.columns(n_cols, gap="medium")
        for i, row in enumerate(df.itertuples(index=False)):
            item_id = str(row.id)
            img_path = ROOT / str(row.image_path)
            with cols[i % n_cols]:
                st.markdown('<div class="product-card catalog-tile">', unsafe_allow_html=True)
                if img_path.exists():
                    _show_photo(img_path, max_width=720, aspect=(3, 4))
                st.markdown(
                    f"""
                    <div class="product-title">{row.title}</div>
                    <div class="product-meta">{row.category} · {row.color}</div>
                    """,
                    unsafe_allow_html=True,
                )
                saved = item_id in wish
                with st.container(horizontal=True, gap="small"):
                    if st.button("Try in Studio", key=f"cat_try_{item_id}", width="stretch"):
                        _go_studio_with_garment(img_path, str(row.title), str(row.category))
                    label = "Saved" if saved else "Save"
                    st.button(
                        label,
                        key=f"cat_wish_{item_id}",
                        width="stretch",
                        on_click=_save_piece if not saved else _remove_piece,
                        args=(item_id, str(row.title)),
                    )
                    st.link_button("Buy", str(row.shop_url), width="stretch")
                st.markdown("</div>", unsafe_allow_html=True)


def _stylist_more(avatar: Image.Image, analysis: dict) -> list:
    more = catalog_for_avatar(
        avatar,
        analysis,
        k=5,
        exclude_ids=st.session_state.get("stylist_shown_ids") or [],
    )
    _remember_recs(more)
    return more


@st.dialog("Upload an avatar")
def _avatar_needed_dialog() -> None:
    st.warning("Please upload an avatar photo first.")
    st.write(
        "The stylist needs a photo to analyze your colors and body type, "
        "and to chat with personal recommendations. Use a front-facing portrait or full-body shot."
    )
    if st.button("Got it", type="primary", width="stretch"):
        st.rerun()


def _images_from_chat_files(files) -> list[Image.Image]:
    images: list[Image.Image] = []
    for uploaded in files or []:
        name = str(getattr(uploaded, "name", "") or "").lower()
        mime = str(getattr(uploaded, "type", "") or "").lower()
        if not (mime.startswith("image/") or name.endswith((".jpg", ".jpeg", ".png", ".webp"))):
            continue
        try:
            images.append(load_rgb(Image.open(uploaded)))
        except Exception:
            continue
    return images


def _transcribe_audio(audio, mime_type: str = "audio/wav"):
    """Load voice transcription from llm_advisor, reloading if Streamlit cached an older module."""
    import importlib

    import src.llm_advisor as advisor

    fn = getattr(advisor, "transcribe_audio", None)
    if fn is None:
        advisor = importlib.reload(advisor)
        fn = getattr(advisor, "transcribe_audio", None)
    if fn is None:
        return "", False
    return fn(audio, mime_type)


def _handle_stylist_prompt(
    prompt: str,
    avatar: Image.Image | None,
    analysis: dict | None,
    *,
    files=None,
    audio=None,
) -> None:
    analysis = analysis or {}
    images = _images_from_chat_files(files)
    audio_bytes = None
    audio_mime = "audio/wav"
    if audio is not None:
        audio_bytes = audio.getvalue() if hasattr(audio, "getvalue") else bytes(audio)
        audio_mime = str(getattr(audio, "type", None) or "audio/wav")
        transcript, _ = _transcribe_audio(audio_bytes, mime_type=audio_mime)
        if transcript and not prompt:
            prompt = transcript
        elif transcript:
            prompt = f"{prompt}\n\n(Voice: {transcript})"
        elif not prompt:
            prompt = "Please style me from this voice note and any attached photos."
    if not prompt and images:
        prompt = "Please analyze this look and recommend what to wear with it, or a better catalog alternative."

    messages = st.session_state.setdefault("stylist_messages", [])
    messages.append(
        {
            "role": "user",
            "content": prompt,
            "images": images,
            "audio": audio_bytes,
            "audio_mime": audio_mime,
        }
    )
    recs = st.session_state.get("stylist_all_recs") or st.session_state.get("stylist_recs") or []
    intent = interpret_chat(prompt, recs)
    new_recs: list = []
    studio_item = None
    search_image = images[0] if images else avatar

    if images:
        has_profile = bool(
            analysis.get("color_season") or analysis.get("body_type") or analysis.get("palette")
        )
        has_dept = resolve_presentation("", analysis) in ("man", "woman")
        if not has_profile or not has_dept:
            try:
                look_analysis = analyze_avatar(images[0])
                if not has_profile:
                    analysis = look_analysis
                    st.session_state["stylist_analysis"] = analysis
                elif resolve_presentation("", analysis) not in ("man", "woman"):
                    look_pres = look_analysis.get("presentation")
                    if look_pres in ("man", "woman"):
                        analysis = {**analysis, "presentation": look_pres}
            except Exception:
                pass

    if intent["action"] == "more" and search_image is not None:
        try:
            new_recs = _stylist_more(search_image, analysis)
        except Exception:
            new_recs = []
    elif intent["action"] == "studio":
        studio_item = intent.get("item")
    elif intent["action"] == "query":
        try:
            new_recs = catalog_for_query(
                intent.get("query") or prompt,
                analysis,
                k=5,
                exclude_ids=st.session_state.get("stylist_shown_ids") or [],
                image=search_image,
            )
        except Exception:
            new_recs = []
        if not new_recs and search_image is not None:
            try:
                new_recs = catalog_for_avatar(
                    search_image,
                    analysis,
                    k=5,
                    exclude_ids=st.session_state.get("stylist_shown_ids") or [],
                )
            except Exception:
                new_recs = []
        if new_recs:
            _remember_recs(new_recs)
    elif images and search_image is not None and not recs:
        try:
            new_recs = catalog_for_avatar(
                search_image,
                analysis,
                k=5,
                exclude_ids=st.session_state.get("stylist_shown_ids") or [],
            )
        except Exception:
            new_recs = []
        if new_recs:
            _remember_recs(new_recs)

    table = new_recs or recs
    if intent["action"] == "colors":
        table = []
    reply, used = reply_to_shopper(
        prompt,
        analysis=analysis,
        recs=table,
        history=messages,
        images=images,
        audio_bytes=audio_bytes,
        audio_mime=audio_mime,
    )
    if intent["action"] == "studio" and studio_item is not None:
        reply = f"{reply}\n\nOpening Studio with {studio_item['title']}."
    messages.append({"role": "assistant", "content": reply, "recs": new_recs, "used_gemini": used})
    if studio_item is not None:
        _go_studio_with_garment(
            ROOT / studio_item["image_path"],
            str(studio_item["title"]),
            str(studio_item.get("category") or ""),
        )


def stylist_tab():
    _init_stylist_state()
    _section_head(
        "Personal atelier",
        "Stylist AI",
        "Upload an avatar for a body-tone reading, then get a color set and catalog looks.",
    )

    vis, form = st.columns([1.05, 1], gap="large")
    with vis:
        avatar = st.session_state.get("stylist_avatar")
        if avatar is not None:
            st.image(_array_jpeg(avatar, max_width=720), caption="Your avatar", width="stretch")
        else:
            stylist_ref = _look_preview_path("screen3_crop.png") or _look_path("screen3_crop.png")
            if stylist_ref:
                _show_photo(stylist_ref, max_width=720)
            st.caption("Full-body, front-facing photos give the clearest color and silhouette read.")
    with form:
        uploaded = st.file_uploader(
            "Avatar photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="stylist_upload",
            help="A front-facing portrait or full-body shot works best.",
        )
        if uploaded is not None:
            avatar = load_rgb(Image.open(uploaded))
            st.session_state["stylist_avatar"] = avatar
        elif avatar is None and st.session_state.get("tryon_person") is not None:
            st.caption("No avatar yet — you can reuse the last Studio subject photo.")
            if st.button("Use last Studio photo", type="secondary"):
                st.session_state["stylist_avatar"] = st.session_state["tryon_person"]
                st.rerun()

        presentation = st.selectbox(
            "Shop for",
            ["Auto (from photo)", "Woman", "Man"],
            index=0,
            key="stylist_shop_for",
            help="Auto reads menswear vs womenswear from the avatar. Override if the read is wrong.",
        )
        analyze = st.button("Analyze", type="primary", width="stretch")
        if avatar is None:
            st.caption("Upload an avatar to unlock body-tone analysis, a color set, and recommendations.")

    avatar = st.session_state.get("stylist_avatar")
    if analyze:
        if avatar is None:
            _avatar_needed_dialog()
            return
        with st.spinner("Reading body tone and recommending a color set…"):
            try:
                try:
                    label_map = avatar_label_map(avatar)
                except Exception:
                    label_map = None
                analysis = analyze_avatar(avatar, label_map=label_map)
                chosen = resolve_presentation(presentation, analysis)
                if chosen in ("man", "woman"):
                    analysis["presentation"] = chosen
                recs = catalog_for_avatar(avatar, analysis, k=5)
            except Exception as exc:
                st.error(f"Stylist failed: {exc}")
                return
        st.session_state["stylist_analysis"] = analysis
        st.session_state["stylist_label_map"] = label_map
        _refresh_stylist_boards(avatar, analysis, chosen if chosen in ("man", "woman") else presentation)
        st.session_state["stylist_shown_ids"] = []
        st.session_state["stylist_all_recs"] = []
        _remember_recs(recs)
        season = analysis.get("color_season") or "your palette"
        undertone = analysis.get("undertone") or "neutral"
        body = analysis.get("body_type") or "silhouette"
        palette = [str(c) for c in (analysis.get("palette") or []) if str(c).strip()]
        palette_line = ", ".join(palette[:5]) if palette else "your recommended set"
        dept = analysis.get("presentation")
        dept_line = (
            "Shopping menswear. "
            if dept == "man"
            else "Shopping womenswear. "
            if dept == "woman"
            else ""
        )
        st.session_state["stylist_messages"] = [
            {
                "role": "assistant",
                "content": (
                    f"{dept_line}Body tone reads {undertone} ({season}) with a {body} frame. "
                    f"A suitable color set is {palette_line}. "
                    "Ask for more recommendations, a weekend outfit, or say “try the top match in Studio.”"
                ),
            }
        ]

    analysis = st.session_state.get("stylist_analysis")
    if analysis and avatar is not None:
        board_style = resolve_presentation(presentation, analysis)
        if st.session_state.get("stylist_board_presentation") != board_style:
            _refresh_stylist_boards(avatar, analysis, board_style)

    if analysis:
        season = str(analysis.get("color_season") or "Unspecified")
        undertone = str(analysis.get("undertone") or "neutral")
        palette = list(analysis.get("palette") or [])
        avoid = list(analysis.get("avoid_colors") or [])
        st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)
        a1, a2 = st.columns(2, gap="large")
        with a1:
            with st.container(border=True):
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.markdown('<p class="analysis-kicker">01 // Body tone</p>', unsafe_allow_html=True)
                st.markdown("**Color analysis**")
                season_pill = _pill(season, "ok")
                undertone_pill = _pill(f"{undertone} undertone", "neutral")
                st.markdown(
                    f"{season_pill} {undertone_pill}",
                    unsafe_allow_html=True,
                )
                st.write(analysis.get("color_notes") or "")
                color_board = st.session_state.get("stylist_color_board")
                if color_board is not None:
                    st.image(
                        color_board,
                        caption="Skin, hair, and cloth sampled from this photo",
                        width="stretch",
                    )
                st.markdown("</div>", unsafe_allow_html=True)
        with a2:
            with st.container(border=True):
                st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
                st.markdown('<p class="analysis-kicker">02 // Silhouette</p>', unsafe_allow_html=True)
                st.markdown("**Body type & style**")
                dept = resolve_presentation("", analysis)
                dept_pill = (
                    _pill("menswear", "ok")
                    if dept == "man"
                    else _pill("womenswear", "ok")
                    if dept == "woman"
                    else ""
                )
                st.markdown(
                    f"{_pill(str(analysis.get('body_type') or 'unspecified'), 'ok')} {dept_pill}",
                    unsafe_allow_html=True,
                )
                body_copy = compact_body_copy(analysis, max_sentences=3)
                if body_copy:
                    st.write(body_copy)
                occasions = analysis.get("occasions") or []
                if occasions:
                    st.caption("Occasions · " + ", ".join(str(o) for o in occasions))
                body_board = st.session_state.get("stylist_body_board")
                if body_board is not None:
                    st.image(
                        body_board,
                        caption="Estimated from a single photo and clothing mask. Not a body scan.",
                        width="stretch",
                    )
                st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)
        _section_head(
            "Palette",
            "Recommended color set",
            f"Wearable colors for a {html.escape(undertone)} undertone in the {html.escape(season)} family.",
        )
        if palette:
            st.markdown(_color_set_html(palette), unsafe_allow_html=True)
        else:
            st.caption("No color set yet — try Analyze again with a clearer, well-lit photo.")
        if avoid:
            st.caption("Ease off")
            st.markdown(_palette_html(avoid), unsafe_allow_html=True)

        st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)
        _section_head(
            "Styled for you",
            "Try these looks",
            "Catalog pieces ranked to your avatar, clothing department, body type, and color set. Send any piece to Studio.",
        )
        _render_stylist_looks(st.session_state.get("stylist_recs") or [], key_prefix="sty_try")
        more_col, studio_col = st.columns(2)
        with more_col:
            if st.button("More recommendations", type="secondary", width="stretch"):
                if avatar is not None:
                    with st.spinner("Finding another edit…"):
                        extra = _stylist_more(avatar, analysis)
                    if extra:
                        st.session_state["stylist_messages"].append(
                            {
                                "role": "assistant",
                                "content": "Another five pieces from the catalog — try one in Studio or keep chatting.",
                                "recs": extra,
                            }
                        )
                        st.rerun()
                    else:
                        st.info("No further catalog matches on this pass. Ask the chatbot for a color or occasion.")
        with studio_col:
            if st.button("Open AI Studio", type="primary", width="stretch"):
                recs = st.session_state.get("stylist_recs") or []
                if recs:
                    _go_studio_with_garment(
                        ROOT / recs[0]["image_path"],
                        str(recs[0]["title"]),
                        str(recs[0].get("category") or ""),
                    )
                else:
                    studio = st.session_state.get("page_studio")
                    if studio is not None:
                        st.switch_page(studio)

    st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)
    _section_head(
        "Ask the atelier",
        "Stylist chat",
        "Gemini stylist with high-fashion sense. Type, speak, or attach a look for recommendations.",
    )
    # Nested so the input stays in the page column (same width as the sections above)
    # instead of a full-bleed bar at the bottom of the browser.
    with st.container(border=True):
        messages = st.session_state.get("stylist_messages") or []
        if not messages:
            st.caption("Chat anytime — attach a photo, record a voice note, or tap Analyze first for a personal reading.")
        for i, msg in enumerate(messages):
            role = msg.get("role", "assistant")
            avatar_icon = ":material/person:" if role == "user" else ":material/auto_awesome:"
            with st.chat_message(role, avatar=avatar_icon):
                extra_images = msg.get("images") or []
                if extra_images:
                    cols = st.columns(min(3, len(extra_images)))
                    for col, look in zip(cols, extra_images):
                        with col:
                            st.image(_array_jpeg(look, max_width=420), width="stretch")
                if msg.get("audio"):
                    st.audio(msg["audio"], format=str(msg.get("audio_mime") or "audio/wav"))
                st.write(msg.get("content", ""))
                extra = msg.get("recs") or []
                if extra:
                    _render_stylist_looks(extra, key_prefix=f"chat_{i}")

        if not any(m.get("role") == "user" for m in messages):
            selected = st.pills(
                "Try asking",
                [
                    "What colors suit me?",
                    "I want to wear for a fashion show",
                    "Casual weekend outfits",
                    "Give me more recommendations",
                ],
                label_visibility="collapsed",
            )
            if selected:
                with st.spinner("Stylist is thinking…"):
                    _handle_stylist_prompt(selected, avatar, analysis)
                st.rerun()

        chat = st.chat_input(
            "Ask, speak, or attach a look…",
            key="stylist_chat_input",
            accept_file="multiple",
            file_type=["jpg", "jpeg", "png", "webp"],
            accept_audio=True,
        )
        if chat:
            text = (chat.text or "").strip()
            files = list(chat.files or [])
            audio = chat.audio
            if text or files or audio is not None:
                with st.spinner("Stylist is thinking…"):
                    _handle_stylist_prompt(text, avatar, analysis, files=files, audio=audio)
                st.rerun()


def profile_tab():
    _section_head(
        "Atelier account",
        "Profile",
        "Session preferences, last try-on, and pieces you saved from Catalog.",
    )

    with st.container(border=True):
        st.markdown("**Identity**")
        st.text_input("Display name", placeholder="e.g. Alex", key="profile_name")
        st.selectbox("Preferred garment region", ["upper", "lower", "dress"], key="pref_region")
        st.caption("Studio uses this as the default region.")

    scores = st.session_state.get("tryon_scores")
    result = st.session_state.get("tryon_result")
    st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)
    _section_head("Archive", "Last Studio render")
    if result is not None:
        r1, r2 = st.columns(2)
        person_prev = st.session_state.get("tryon_person")
        if person_prev is not None:
            _show_array(person_prev, caption="Original", slot=r1)
        _show_array(result, caption="AI render", slot=r2)
        if scores:
            with st.container(border=True):
                st.markdown(
                    _score_bar("Try-on confidence", scores["tryon_conf"], 0.85),
                    unsafe_allow_html=True,
                )
        if st.button("Open Studio", type="primary"):
            studio = st.session_state.get("page_studio")
            if studio is not None:
                st.switch_page(studio)
    else:
        st.info("No try-on in this session yet. Generate one in Studio.")

    st.markdown('<hr class="hr-rule" />', unsafe_allow_html=True)
    _section_head("Wishlist", "Saved pieces", "Compact bookmarks from Catalog. Tap Remove to drop a piece.")
    _flush_wish_notice()
    _render_saved_pieces(key_prefix="wish_rm", compact=True)


def _render_topbar(home, studio, catalog, stylist, profile) -> None:
    brand, n1, n2, n3, n4, n5 = st.columns([2.1, 1, 1, 1, 1.25, 1], vertical_alignment="center")
    with brand:
        logo_uri = _nav_logo_uri()
        logo_html = f'<img src="{logo_uri}" alt="VESTURE" />' if logo_uri else ""
        st.markdown(
            f'<div class="nav-brand">{logo_html}'
            '<div class="brand-mark" style="font-size:1.55rem; margin:0;">VESTURE</div></div>',
            unsafe_allow_html=True,
        )
    with n1:
        st.page_link(home, label="Home", icon=":material/home:", width="stretch")
    with n2:
        st.page_link(studio, label="Studio", icon=":material/checkroom:", width="stretch")
    with n3:
        st.page_link(catalog, label="Catalog", icon=":material/apparel:", width="stretch")
    with n4:
        st.page_link(stylist, label="Stylist AI", icon=":material/auto_awesome:", width="stretch")
    with n5:
        st.page_link(profile, label="Profile", icon=":material/person:", width="stretch")
    st.markdown('<hr class="hr-rule" style="margin:.5rem 0 1.8rem 0;" />', unsafe_allow_html=True)


def main():
    home = st.Page(home_tab, title="Home", icon=":material/home:", url_path="home", default=True)
    studio = st.Page(tryon_tab, title="Studio", icon=":material/checkroom:", url_path="studio")
    catalog = st.Page(catalog_tab, title="Catalog", icon=":material/apparel:", url_path="catalog")
    stylist = st.Page(stylist_tab, title="Stylist AI", icon=":material/auto_awesome:", url_path="stylist")
    profile = st.Page(profile_tab, title="Profile", icon=":material/person:", url_path="profile")

    st.session_state["page_studio"] = studio
    st.session_state["page_catalog"] = catalog
    st.session_state["page_stylist"] = stylist

    page = st.navigation([home, studio, catalog, stylist, profile], position="hidden")
    _inject_css()
    _render_topbar(home, studio, catalog, stylist, profile)
    page.run()

    st.markdown(
        '<div class="atelier-footer">© 2026 VESTURE Digital Atelier · SegFormer · IDM-VTON · FashionCLIP · Gemini</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
