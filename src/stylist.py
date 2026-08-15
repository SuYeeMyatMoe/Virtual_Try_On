"""Avatar color / body analysis and stylist chat intents."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .llm_advisor import analyze_avatar_llm, has_gemini, stylist_chat
from .recommend import recommend_from_text, recommend_top_k
from .segmentation import load_segformer, segment_clothing

SWATCH_HEX = {
    "black": "#111111", "white": "#F4F1EC", "ivory": "#F3E6D0", "cream": "#E8D5B5",
    "navy": "#1B2A4A", "navy blue": "#1B2A4A", "charcoal": "#3A3A3A", "grey": "#8A8580",
    "gray": "#8A8580", "olive": "#5C5A32", "camel": "#C4A574", "rust": "#B85C38",
    "peach": "#E8B4A0", "coral": "#D9786A", "rose": "#C9899B", "lavender": "#A89BB8",
    "powder blue": "#9BB5C8", "emerald": "#1F6B4A", "true red": "#B42318",
    "chocolate brown": "#5A3825", "brown": "#6B4423", "green": "#3D6B4F",
    "violet": "#8B5CF6", "purple": "#6D3BD7", "pink": "#D4A0B0", "leaf green": "#5B7F3A",
    "warm camel": "#C4A574", "soft grey": "#A39E99", "stark black": "#0A0A0A",
    "icy grey": "#C9CDD1", "icy blue": "#C5D4E0", "hot pink": "#D63B7A",
    "orange": "#D9762C", "mustard": "#C4A035", "muted olive": "#6B6A3D",
    "pure white": "#FFFFFF", "cool fuchsia": "#C23B86", "warm brown": "#6B4423",
}

BOARD_BG = (18, 16, 20)
BOARD_INK = (244, 241, 236)
BOARD_SOFT = (180, 170, 160)
BOARD_FAINT = (140, 130, 120)
BOARD_GOLD = (196, 165, 116)
BOARD_FILL = (40, 36, 42)
BOARD_SIZE = (1100, 620)
BANDS = (
    ("shoulders", 0.22, 0.38, (139, 92, 246, 55)),
    ("waist", 0.42, 0.55, (16, 185, 129, 50)),
    ("hips", 0.58, 0.78, (208, 188, 255, 45)),
)

# Geometric icons. Woman = softer outline; Man = broader shoulders / straighter hem.
_SILHOUETTES: dict[str, dict[str, list[tuple[int, int]]]] = {
    "neutral": {
        "hourglass": [(80, 40), (140, 40), (160, 90), (100, 140), (160, 220), (40, 220), (100, 140), (40, 90)],
        "pear": [(90, 40), (130, 40), (145, 100), (170, 230), (30, 230), (55, 100)],
        "inverted triangle": [(40, 40), (180, 40), (150, 100), (130, 230), (70, 230), (50, 100)],
        "rectangle": [(70, 40), (150, 40), (150, 230), (70, 230)],
        "athletic": [(75, 40), (145, 40), (145, 230), (75, 230)],
        "apple": [(70, 40), (150, 40), (175, 130), (155, 230), (65, 230), (45, 130)],
    },
    "woman": {
        "hourglass": [(88, 36), (132, 36), (158, 95), (100, 145), (158, 228), (42, 228), (100, 145), (42, 95)],
        "pear": [(96, 38), (124, 38), (140, 105), (172, 232), (28, 232), (60, 105)],
        "inverted triangle": [(50, 38), (170, 38), (148, 105), (128, 228), (72, 228), (52, 105)],
        "rectangle": [(78, 38), (142, 38), (146, 228), (74, 228)],
        "athletic": [(82, 38), (138, 38), (142, 228), (78, 228)],
        "apple": [(78, 38), (142, 38), (172, 135), (150, 228), (70, 228), (48, 135)],
    },
    "man": {
        "hourglass": [(62, 36), (158, 36), (168, 88), (108, 150), (162, 228), (38, 228), (92, 150), (32, 88)],
        "pear": [(78, 36), (142, 36), (152, 100), (168, 228), (32, 228), (48, 100)],
        "inverted triangle": [(28, 36), (192, 36), (158, 100), (138, 228), (62, 228), (42, 100)],
        "rectangle": [(58, 36), (162, 36), (158, 228), (62, 228)],
        "athletic": [(52, 36), (168, 36), (162, 228), (58, 228)],
        "apple": [(58, 36), (162, 36), (178, 128), (160, 228), (60, 228), (42, 128)],
    },
}

MORE_PHRASES = (
    "more rec",
    "more look",
    "more option",
    "more piece",
    "another",
    "show more",
    "other option",
    "more recommend",
    "give me more",
    "next set",
)
STUDIO_PHRASES = (
    "try in studio",
    "try in ai studio",
    "try on",
    "send to studio",
    "open studio",
    "wear this",
    "try the",
    "try that",
)


def _mean_rgb(pixels: np.ndarray) -> Optional[np.ndarray]:
    if pixels.size == 0:
        return None
    return pixels.reshape(-1, 3).mean(axis=0)


def _skin_rgb(arr: np.ndarray, label_map: np.ndarray) -> Optional[np.ndarray]:
    """Mean RGB of exposed skin: face, then arms if present."""
    chunks = []
    for class_id in (11, 14, 15):
        pix = arr[label_map == class_id]
        if pix.size:
            chunks.append(pix.reshape(-1, 3))
    if not chunks:
        return None
    return np.vstack(chunks).mean(axis=0)


def _season_from_rgb(rgb: np.ndarray) -> tuple[str, str, list[str], list[str], str]:
    r, g, b = [float(x) for x in rgb]
    warmth = r - b
    value = (r + g + b) / 3.0
    undertone = "warm" if warmth > 12 else "cool" if warmth < -12 else "neutral"
    deep = value < 118
    if undertone == "warm" and not deep:
        season, palette, avoid = (
            "Light Spring",
            ["ivory", "peach", "coral", "warm camel", "leaf green"],
            ["stark black", "icy grey", "cool fuchsia"],
        )
    elif undertone == "warm":
        season, palette, avoid = (
            "Soft Autumn",
            ["camel", "olive", "rust", "cream", "chocolate brown"],
            ["pure white", "icy blue", "hot pink"],
        )
    elif undertone == "cool" and not deep:
        season, palette, avoid = (
            "Light Summer",
            ["powder blue", "lavender", "rose", "soft grey", "navy"],
            ["orange", "mustard", "warm brown"],
        )
    else:
        season, palette, avoid = (
            "True Winter",
            ["black", "white", "navy", "emerald", "true red"],
            ["camel", "orange", "muted olive"],
        )
    notes = (
        f"Body tone reads {undertone} with {'deeper' if deep else 'lighter'} value. "
        f"The {season} color set below will sit cleanly against this skin."
    )
    return season, undertone, palette, avoid, notes


def _body_from_labels(label_map: np.ndarray) -> tuple[str, str]:
    h, w = label_map.shape
    if h < 8 or w < 8:
        return "unspecified", "The crop is too tight to estimate silhouette. Use a full-body, front-facing photo."

    def width_at(labels: list[int], y0: float, y1: float) -> float:
        ys0, ys1 = int(h * y0), max(int(h * y1), int(h * y0) + 1)
        band = label_map[ys0:ys1]
        mask = np.isin(band, labels)
        if not mask.any():
            return 0.0
        cols = np.where(mask.any(axis=0))[0]
        if len(cols) < 2:
            return 0.0
        return float(cols[-1] - cols[0]) / max(w, 1)

    shoulders = width_at([4, 14, 15, 7], 0.22, 0.38)
    waist = width_at([4, 7, 6], 0.42, 0.55)
    hips = width_at([5, 6, 12, 13], 0.58, 0.78)
    has_legs = np.isin(label_map, [12, 13, 5, 6]).mean() > 0.04
    has_face = np.isin(label_map, [11]).mean() > 0.005
    if not has_legs or not has_face:
        return (
            "unspecified",
            "This looks like a crop rather than a full-body portrait. "
            "Upload a front-facing full-length photo for a clearer body-type read.",
        )
    if shoulders and hips and shoulders > hips * 1.12:
        return (
            "inverted triangle",
            "Shoulder line reads broader than the hip line. Ease the upper half "
            "and add presence on the lower half.",
        )
    if shoulders and hips and hips > shoulders * 1.12:
        return (
            "pear",
            "Hip line reads wider than the shoulders. Add structure at the top "
            "and keep the lower half in a continuous mid or dark tone.",
        )
    if waist and shoulders and hips and waist > max(shoulders, hips) * 1.05:
        return (
            "apple",
            "The midsection reads fuller than shoulder and hip. A longer line through "
            "the middle keeps proportion calm.",
        )
    if waist and shoulders and hips and waist < min(shoulders, hips) * 0.86:
        return (
            "hourglass",
            "Waist reads narrower than shoulder and hip. A defined waist or a jacket "
            "that nips in will follow the frame.",
        )
    if shoulders and hips and abs(shoulders - hips) < 0.08:
        return (
            "rectangle",
            "Shoulder and hip widths are close. Layering or a break at the waist adds shape.",
        )
    return (
        "athletic",
        "Proportions look balanced and straight. Tailored layers and a clear hem line add interest.",
    )


def align_label_map(label_map: np.ndarray, image: Image.Image) -> np.ndarray:
    """Resize SegFormer labels to the avatar with nearest-neighbor."""
    w, h = image.size
    if label_map.shape == (h, w):
        return label_map
    return np.array(
        Image.fromarray(label_map.astype(np.uint8)).resize((w, h), Image.Resampling.NEAREST)
    )


def avatar_label_map(image: Image.Image) -> np.ndarray:
    """One SegFormer pass for analysis boards and fallback geometry."""
    load_segformer()
    _, _, label_map = segment_clothing(image, category="upper")
    return align_label_map(label_map, image)


def fallback_analysis(image: Image.Image, label_map: Optional[np.ndarray] = None) -> dict:
    """SegFormer geometry + face color when Gemini is unavailable."""
    try:
        if label_map is None:
            label_map = avatar_label_map(image)
        else:
            label_map = align_label_map(label_map, image)
        arr = np.asarray(image.convert("RGB"))
        rgb = _skin_rgb(arr, label_map)
        if rgb is None:
            rgb = _mean_rgb(arr.reshape(-1, 3)[::80])
        season, undertone, palette, avoid, color_notes = _season_from_rgb(rgb if rgb is not None else np.array([140, 110, 95]))
        body_type, body_notes = _body_from_labels(label_map)
    except Exception:
        season, undertone, palette, avoid, color_notes = _season_from_rgb(np.array([140, 110, 95]))
        body_type, body_notes = (
            "unspecified",
            "Could not read silhouette from this photo. Try a brighter, full-body shot.",
        )
    return {
        "color_season": season,
        "undertone": undertone,
        "palette": palette,
        "avoid_colors": avoid,
        "color_notes": color_notes,
        "body_type": body_type,
        "body_notes": body_notes,
        "style_direction": (
            f"Start from the {season} color set and a {body_type} silhouette: "
            "one tailored layer, one easy piece, and shoes that stay in the same value family."
        ),
        "silhouette_tips": body_notes,
        "occasions": ["weekday", "smart casual"],
        "used_gemini": False,
    }


def analyze_avatar(image: Image.Image, label_map: Optional[np.ndarray] = None) -> dict:
    payload, used = analyze_avatar_llm(image)
    complete = bool(
        used and payload.get("color_season") and payload.get("body_type") and payload.get("palette")
    )
    if complete:
        payload["used_gemini"] = True
        payload.setdefault("avoid_colors", [])
        payload.setdefault("occasions", [])
        return payload
    fallback = fallback_analysis(image, label_map=label_map)
    if used and payload.get("color_season") and payload.get("body_type"):
        payload["used_gemini"] = True
        if not payload.get("palette"):
            payload["palette"] = fallback.get("palette") or []
        if not payload.get("avoid_colors"):
            payload["avoid_colors"] = fallback.get("avoid_colors") or []
        payload.setdefault("occasions", [])
        if not payload.get("color_notes"):
            payload["color_notes"] = fallback.get("color_notes")
        return payload
    if payload:
        fallback.update({k: v for k, v in payload.items() if v not in (None, "", [], {})})
        fallback["used_gemini"] = used
    return fallback


def palette_hints(analysis: Optional[dict]) -> list[str]:
    if not analysis:
        return []
    hints = list(analysis.get("palette") or [])
    season = str(analysis.get("color_season") or "")
    hints.extend(season.replace("-", " ").split())
    return [h for h in hints if h]


def catalog_for_avatar(
    image: Image.Image,
    analysis: Optional[dict] = None,
    k: int = 5,
    exclude_ids: Optional[Sequence] = None,
    offset: int = 0,
) -> list[dict]:
    return recommend_top_k(
        image,
        k=k,
        exclude_ids=exclude_ids,
        offset=offset,
        color_hints=palette_hints(analysis),
    )


def catalog_for_query(
    query: str,
    analysis: Optional[dict] = None,
    k: int = 5,
    exclude_ids: Optional[Sequence] = None,
) -> list[dict]:
    return recommend_from_text(
        query,
        k=k,
        exclude_ids=exclude_ids,
        color_hints=palette_hints(analysis),
    )


def match_catalog_item(text: str, recs: Sequence[dict]) -> Optional[dict]:
    t = text.lower()
    if any(p in t for p in ("top match", "first", "the first", "number 1", "no. 1")):
        return recs[0] if recs else None
    best, best_n = None, 0
    for item in recs:
        title = str(item.get("title", "")).lower()
        words = [w for w in title.replace("-", " ").split() if len(w) > 2]
        hits = sum(1 for w in words if w in t)
        if title and title in t:
            hits += 3
        if hits > best_n:
            best, best_n = item, hits
    return best if best_n >= 2 else None


def interpret_chat(text: str, recs: Sequence[dict]) -> dict[str, Any]:
    t = text.lower().strip()
    wants_more = any(p in t for p in MORE_PHRASES)
    wants_studio = any(p in t for p in STUDIO_PHRASES)
    item = match_catalog_item(text, recs)
    if wants_studio:
        return {"action": "studio", "item": item or (recs[0] if recs else None)}
    if wants_more:
        return {"action": "more"}
    wants_recs = any(
        p in t
        for p in (
            "recommend",
            "suggest",
            "what should i wear",
            "outfit",
            "what to wear",
            "show me",
        )
    )
    if wants_recs:
        return {"action": "query", "query": text}
    return {"action": "chat"}


def reply_to_shopper(
    text: str,
    *,
    analysis: dict,
    recs: Sequence[dict],
    history: Sequence[dict],
    images: Optional[Sequence[Image.Image]] = None,
    audio_bytes: Optional[bytes] = None,
    audio_mime: str = "audio/wav",
) -> tuple[str, bool]:
    return stylist_chat(
        text,
        history=history,
        analysis=analysis,
        recs=recs,
        images=images,
        audio_bytes=audio_bytes,
        audio_mime=audio_mime,
    )


def gemini_ready() -> bool:
    return has_gemini()


def swatch_hex(name: str) -> str:
    key = str(name or "").lower()
    for name_key, hex_v in SWATCH_HEX.items():
        if name_key in key:
            return hex_v
    return "#8B5CF6"


def _hex_rgb(hex_v: str) -> tuple[int, int, int]:
    h = str(hex_v).lstrip("#")
    if len(h) != 6:
        return (139, 92, 246)
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def _board_font(size: int = 28) -> ImageFont.ImageFont:
    for path in (
        "arial.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _mean_rgb_of(arr: np.ndarray, label_map: Optional[np.ndarray], class_id: int) -> Optional[tuple[int, int, int]]:
    if label_map is None or label_map.shape[:2] != arr.shape[:2]:
        return None
    pix = arr[label_map == class_id]
    if pix.size == 0:
        return None
    mean = pix.reshape(-1, 3).mean(axis=0)
    return tuple(int(x) for x in mean)


def _fit_thumb(avatar: Image.Image, max_size: tuple[int, int] = (420, 560)) -> Image.Image:
    thumb = avatar.convert("RGB").copy()
    thumb.thumbnail(max_size, Image.Resampling.LANCZOS)
    return thumb


def _normalize_presentation(presentation: str) -> str:
    key = str(presentation or "neutral").strip().lower()
    if key in {"woman", "women", "female"}:
        return "woman"
    if key in {"man", "men", "male"}:
        return "man"
    return "neutral"


def _normalize_body_type(body_type: str) -> str:
    key = str(body_type or "unspecified").strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "inverted triangle": "inverted triangle",
        "invertedtriangle": "inverted triangle",
        "hour glass": "hourglass",
        "hourglass": "hourglass",
        "pear": "pear",
        "apple": "apple",
        "rectangle": "rectangle",
        "athletic": "athletic",
        "unspecified": "unspecified",
    }
    return aliases.get(key, key if key in _SILHOUETTES["neutral"] else "unspecified")


def _dashed_rect(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill, dash: int = 10, width: int = 2) -> None:
    x0, y0, x1, y1 = box
    segments = [
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ]
    for (ax, ay), (bx, by) in segments:
        length = max(abs(bx - ax), abs(by - ay), 1)
        steps = max(1, length // dash)
        for i in range(steps):
            if i % 2:
                continue
            t0 = i / steps
            t1 = min(1.0, (i + 1) / steps)
            draw.line(
                (ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1),
                fill=fill,
                width=width,
            )


def _wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = str(text or "").split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    dummy = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    for word in words[1:]:
        trial = f"{current} {word}"
        bbox = dummy.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines[:6]


def _overlay_bands(board: Image.Image, box: tuple[int, int, int, int], label_map: Optional[np.ndarray]) -> Image.Image:
    if label_map is None:
        return board
    x, y, w, h = box
    overlay = Image.new("RGBA", board.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _board_font(14)
    for name, y0, y1, fill in BANDS:
        top = y + int(h * y0)
        bot = y + int(h * y1)
        draw.rectangle((x, top, x + w, bot), fill=fill)
        draw.text((x + 8, top + 4), name, fill=(244, 241, 236, 220), font=font)
    return Image.alpha_composite(board.convert("RGBA"), overlay).convert("RGB")


def build_color_board(
    avatar: Image.Image,
    analysis: dict,
    label_map: Optional[np.ndarray] = None,
) -> Image.Image:
    avatar = avatar.convert("RGB")
    if label_map is not None:
        label_map = align_label_map(label_map, avatar)
    arr = np.asarray(avatar)
    skin = _mean_rgb_of(arr, label_map, 11) or (180, 140, 120)
    hair = _mean_rgb_of(arr, label_map, 2) or (40, 30, 25)
    cloth = _mean_rgb_of(arr, label_map, 4)

    W, H = BOARD_SIZE
    board = Image.new("RGB", (W, H), BOARD_BG)
    draw = ImageDraw.Draw(board)
    title_font = _board_font(32)
    sub_font = _board_font(20)
    small_font = _board_font(16)

    thumb = _fit_thumb(avatar)
    ty = (H - thumb.height) // 2
    board.paste(thumb, (40, ty))

    x = 500
    season = str(analysis.get("color_season") or "Unspecified")
    undertone = str(analysis.get("undertone") or "neutral")
    draw.text((x, 36), season, fill=BOARD_INK, font=title_font)
    draw.text((x, 86), f"{undertone} undertone", fill=BOARD_SOFT, font=sub_font)

    samples = [("skin", skin), ("hair", hair)]
    if cloth is not None:
        samples.append(("cloth", cloth))
    for i, (name, rgb) in enumerate(samples):
        cx = x + 40 + i * 90
        cy = 168
        draw.ellipse((cx - 28, cy - 28, cx + 28, cy + 28), fill=rgb)
        draw.text((cx - 18, cy + 36), name, fill=BOARD_SOFT, font=small_font)

    y = 250
    draw.text((x, y), "Recommended color set", fill=BOARD_FAINT, font=small_font)
    y += 28
    for name in list(analysis.get("palette") or [])[:5]:
        rgb = _hex_rgb(swatch_hex(str(name)))
        draw.rounded_rectangle((x, y, x + 280, y + 44), radius=8, fill=rgb)
        ink = (20, 20, 20) if sum(rgb) > 400 else (255, 255, 255)
        draw.text((x + 12, y + 12), str(name), fill=ink, font=small_font)
        y += 52
    return board


def build_body_board(
    avatar: Image.Image,
    analysis: dict,
    label_map: Optional[np.ndarray] = None,
    presentation: str = "Neutral",
) -> Image.Image:
    avatar = avatar.convert("RGB")
    if label_map is not None:
        label_map = align_label_map(label_map, avatar)
    body = _normalize_body_type(str(analysis.get("body_type") or "unspecified"))
    style = _normalize_presentation(presentation)

    W, H = BOARD_SIZE
    board = Image.new("RGB", (W, H), BOARD_BG)
    title_font = _board_font(32)
    sub_font = _board_font(18)
    small_font = _board_font(16)

    thumb = _fit_thumb(avatar)
    tx, ty = 40, (H - thumb.height) // 2
    board.paste(thumb, (tx, ty))
    board = _overlay_bands(board, (tx, ty, thumb.width, thumb.height), label_map)
    draw = ImageDraw.Draw(board)

    label = body.replace("_", " ")
    draw.text((620, 36), label, fill=BOARD_INK, font=title_font)
    draw.text((620, 82), "Estimated silhouette · not a body scan", fill=BOARD_FAINT, font=sub_font)
    draw.text((620, 108), f"Icon style · {style}", fill=BOARD_FAINT, font=small_font)

    icon_box = (640, 150, 1040, 560)
    if body == "unspecified":
        _dashed_rect(draw, icon_box, fill=BOARD_GOLD, dash=12, width=2)
        notes = str(analysis.get("body_notes") or "Upload a front-facing full-length photo.")
        y = 190
        for line in _wrap_text(notes, small_font, 360):
            draw.text((660, y), line, fill=BOARD_SOFT, font=small_font)
            y += 26
    else:
        family = _SILHOUETTES.get(style) or _SILHOUETTES["neutral"]
        poly = family.get(body) or _SILHOUETTES["neutral"].get(body) or _SILHOUETTES["neutral"]["rectangle"]
        shifted = [(p[0] + 700, p[1] + 180) for p in poly]
        draw.polygon(shifted, outline=BOARD_GOLD, fill=BOARD_FILL)
    return board


def build_analysis_boards(
    avatar: Image.Image,
    analysis: dict,
    label_map: Optional[np.ndarray] = None,
    presentation: str = "Neutral",
) -> tuple[Image.Image, Image.Image]:
    return (
        build_color_board(avatar, analysis, label_map),
        build_body_board(avatar, analysis, label_map, presentation),
    )
