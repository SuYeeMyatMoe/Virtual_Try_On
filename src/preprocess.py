"""Image preprocessing helpers for person and garment inputs."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps


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
    "shoes": "lower",
    "shoe": "lower",
    "heels": "lower",
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


COLOR_ALIASES = {
    "yello": "yellow",
    "yelow": "yellow",
    "yeellow": "yellow",
    "navy blue": "navy",
    "navyblue": "navy",
    "gray": "grey",
    "dark blue": "navy",
    "light blue": "blue",
    "sky blue": "blue",
    "hot pink": "pink",
    "off white": "white",
    "off-white": "white",
    "blond": "blonde",
    "platinum blonde": "platinum",
    "dirty blonde": "blonde",
    "brunette": "brown",
    "ginger": "red",
    "darkbrown": "dark brown",
    "lightbrown": "light brown",
}

COLOR_RGB = {
    "yellow": (240, 200, 40),
    "navy": (25, 55, 109),
    "blue": (50, 110, 190),
    "red": (178, 34, 52),
    "black": (22, 22, 22),
    "white": (245, 245, 245),
    "green": (40, 120, 70),
    "pink": (220, 120, 150),
    "beige": (196, 176, 148),
    "grey": (128, 128, 128),
    "brown": (121, 85, 61),
    "olive": (85, 107, 47),
    "cream": (245, 240, 220),
    "ivory": (255, 255, 240),
    "purple": (110, 60, 160),
    "gold": (196, 165, 80),
    "orange": (220, 120, 40),
    "coral": (255, 127, 80),
    "burgundy": (128, 0, 32),
    "teal": (0, 128, 128),
    "khaki": (189, 166, 120),
    "charcoal": (54, 54, 54),
    "blonde": (216, 180, 108),
    "platinum": (230, 224, 208),
    "auburn": (138, 50, 28),
    "copper": (186, 90, 38),
    "silver": (170, 174, 182),
    "dark brown": (48, 30, 20),
    "light brown": (138, 96, 58),
}


def normalize_color_name(color: Optional[str]) -> Optional[str]:
    """Map typed colors (including typos) onto a short catalog name."""
    raw = " ".join(str(color or "").strip().lower().split())
    if not raw:
        return None
    if raw in COLOR_ALIASES:
        return COLOR_ALIASES[raw]
    if raw in COLOR_RGB:
        return raw
    for name in COLOR_RGB:
        if name in raw or raw in name:
            return name
    return raw


def color_to_rgb(color: Optional[str]) -> Optional[Tuple[int, int, int]]:
    name = normalize_color_name(color)
    if not name:
        return None
    return COLOR_RGB.get(name)


# Hair dyes — distinct enough to read in a before/after, not neon.
HAIR_COLOR_RGB = {
    "black": (22, 16, 14),
    "dark brown": (52, 32, 20),
    "brown": (108, 66, 38),
    "light brown": (150, 108, 70),
    "blonde": (196, 166, 118),
    "platinum": (214, 206, 192),
    "auburn": (128, 54, 32),
    "copper": (168, 92, 48),
    "red": (156, 48, 34),
    "burgundy": (92, 28, 36),
    "pink": (188, 118, 136),
    "blue": (58, 78, 132),
    "silver": (156, 160, 168),
}


def hair_color_to_rgb(color: Optional[str]) -> Optional[Tuple[int, int, int]]:
    name = normalize_color_name(color)
    if not name:
        return None
    if name in HAIR_COLOR_RGB:
        return HAIR_COLOR_RGB[name]
    base = COLOR_RGB.get(name)
    if base is None:
        return None
    lum = 0.299 * base[0] + 0.587 * base[1] + 0.114 * base[2]
    sat = 0.40
    return (
        int(sat * base[0] + (1.0 - sat) * lum),
        int(sat * base[1] + (1.0 - sat) * lum),
        int(sat * base[2] + (1.0 - sat) * lum),
    )


def tint_garment(image: Image.Image, color: Optional[str]) -> Image.Image:
    """Keep fabric texture but shift the garment still toward the requested color."""
    rgb = color_to_rgb(color)
    if image is None or rgb is None:
        return image
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    lum = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]) / 255.0
    tinted = np.stack(
        [rgb[0] * lum, rgb[1] * lum, rgb[2] * lum],
        axis=-1,
    )
    mixed = 0.84 * tinted + 0.16 * arr
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8), mode="RGB")


def recolor_hair(
    image: Image.Image,
    mask: Image.Image,
    color: Optional[str],
) -> Image.Image:
    """Shift hair hue while keeping strand texture, roots, and photo lighting."""
    rgb = hair_color_to_rgb(color)
    if image is None or rgb is None:
        return image
    person = image.convert("RGB")
    alpha_img = mask.convert("L")
    if alpha_img.size != person.size:
        alpha_img = alpha_img.resize(person.size, Image.Resampling.BILINEAR)
    # Soft hairline — a hard mask reads as a sticker.
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=2.2))
    arr = np.asarray(person, dtype=np.float32)
    alpha = np.clip(np.asarray(alpha_img, dtype=np.float32) / 255.0, 0.0, 1.0)
    hair = alpha > 0.06
    if not hair.any():
        return person

    blur = np.asarray(person.filter(ImageFilter.GaussianBlur(radius=1.15)), dtype=np.float32)
    residual = arr - blur

    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    lo, hi = np.percentile(lum[hair], [6, 97])
    lum_n = np.clip((lum - lo) / max(float(hi - lo), 8.0), 0.0, 1.0)
    src_med = float(np.median(lum[hair]))
    tr, tg, tb = (float(rgb[0]), float(rgb[1]), float(rgb[2]))
    t_lum = max(0.299 * tr + 0.587 * tg + 0.114 * tb, 1.0)

    # Going darker (ash/blonde → brown) needs a real drop; going lighter
    # still keeps most photo lighting so it does not turn into a flat fill.
    going_darker = t_lum < src_med - 6.0
    if going_darker:
        gap = (src_med - t_lum) / max(src_med, 1.0)
        lift = float(np.clip(0.62 + 0.30 * gap, 0.62, 0.90))
        local_lift = lift * (0.88 - 0.20 * lum_n)
    else:
        lift = float(np.clip(0.30 + 0.28 * (t_lum - src_med) / 200.0, 0.30, 0.56))
        local_lift = lift * (0.48 + 0.52 * lum_n)
    scaled = lum * (t_lum / max(src_med, 12.0))
    new_lum = np.clip(lum * (1.0 - local_lift) + scaled * local_lift, 0.0, 255.0)

    sat = 0.78 if going_darker else 0.64
    soft_t = (
        sat * tr + (1.0 - sat) * t_lum,
        sat * tg + (1.0 - sat) * t_lum,
        sat * tb + (1.0 - sat) * t_lum,
    )
    colored = np.stack(
        [
            soft_t[0] / t_lum * new_lum,
            soft_t[1] / t_lum * new_lum,
            soft_t[2] / t_lum * new_lum,
        ],
        axis=-1,
    )
    mix = 0.90 if going_darker else 0.80
    dyed = (1.0 - mix) * arr + mix * colored
    dyed = np.clip(dyed + residual * 1.05, 0, 255)

    a = (alpha * (0.96 if going_darker else 0.90))[..., None]
    out = (1.0 - a) * arr + a * dyed
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


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
    named = normalize_color_name(color)
    if named:
        parts.append(named)
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
