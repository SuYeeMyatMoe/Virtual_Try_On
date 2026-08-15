"""Image preprocessing helpers for person and garment inputs."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps


REGION_ALIASES = {
    "upper": "upper",
    "upper body": "upper",
    "top": "upper",
    "tops": "upper",
    "shirt": "upper",
    "blouse": "upper",
    "lower": "lower",
    "lower body": "lower",
    "low": "lower",
    "bottom": "lower",
    "bottoms": "lower",
    "pant": "lower",
    "pants": "lower",
    "jean": "lower",
    "jeans": "lower",
    "skirt": "lower",
    "trousers": "lower",
    "shorts": "lower",
    "dress": "dress",
    "overall": "dress",
    "full": "dress",
    "gown": "dress",
}


def normalize_garment_region(category: str | None) -> str:
    """Map UI / catalog / aliases onto upper | lower | dress."""
    key = str(category or "upper").strip().lower()
    if key in REGION_ALIASES:
        return REGION_ALIASES[key]
    if any(word in key for word in ("pant", "jean", "skirt", "trouser", "short", "bottom")):
        return "lower"
    if "dress" in key or "gown" in key:
        return "dress"
    return "upper"


PERSON_SIZE = (768, 1024)  # width, height
FAST_SIZE = (384, 512)
GARMENT_SIZE = (512, 512)
MIN_SHORT_SIDE = 256


def load_rgb(image: Image.Image) -> Image.Image:
    """EXIF-correct and convert to RGB."""
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def resize_letterbox(
    image: Image.Image,
    size: Tuple[int, int],
    fill: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Resize keeping aspect ratio, pad to exact size."""
    target_w, target_h = size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), fill)
    offset = ((target_w - new_w) // 2, (target_h - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def preprocess_person(
    image: Image.Image,
    fast: bool = False,
) -> Image.Image:
    """Prepare a full-body person photo for segmentation / try-on."""
    image = load_rgb(image)
    size = FAST_SIZE if fast else PERSON_SIZE
    return resize_letterbox(image, size)


def preprocess_garment(image: Image.Image) -> Image.Image:
    """Prepare a garment product image."""
    image = load_rgb(image)
    return resize_letterbox(image, GARMENT_SIZE)


def quality_check(image: Image.Image) -> Tuple[bool, str]:
    """Basic input quality checks. Returns (ok, message)."""
    if image is None:
        return False, "No image provided."
    w, h = image.size
    if min(w, h) < MIN_SHORT_SIDE:
        return False, f"Image too small (min side {MIN_SHORT_SIDE}px required)."
    arr = np.asarray(load_rgb(image))
    if arr.std() < 5.0:
        return False, "Image appears blank or nearly uniform."
    return True, "OK"


def build_garment_prompt(
    category: str,
    color: Optional[str] = None,
    style: Optional[str] = None,
    extra: Optional[str] = None,
) -> str:
    """Template prompt for SD inpainting (no LLM)."""
    parts = []
    if color:
        parts.append(color.strip())
    if style:
        parts.append(style.strip())
    cat = normalize_garment_region(category)
    mapping = {
        "upper": "shirt or blouse on the upper body",
        "lower": "pants or skirt on the legs, not a top",
        "dress": "dress covering the body",
    }
    parts.append(mapping[cat])
    if extra:
        parts.append(extra.strip())
    prompt = ", ".join(p for p in parts if p)
    return (
        f"photorealistic fashion photo of a person wearing {prompt}, "
        "natural lighting, sharp fabric details, realistic wrinkles"
    )


def negative_prompt() -> str:
    return (
        "blurry, deformed body, extra limbs, bad anatomy, watermark, text, "
        "low quality, cartoon, painting, face distortion"
    )
