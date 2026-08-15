"""Avatar color / body analysis and stylist chat intents."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image

from .llm_advisor import analyze_avatar_llm, has_gemini, stylist_chat
from .recommend import recommend_from_text, recommend_top_k
from .segmentation import load_segformer, segment_clothing

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
        f"Skin reading leans {undertone} with {'deeper' if deep else 'lighter'} value. "
        f"Build outfits in {season.lower()} colors and keep contrast clean."
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
            "Shoulders read broader than the hip line. Soften the top with open necklines "
            "and add volume or a lighter color on the lower half.",
        )
    if shoulders and hips and hips > shoulders * 1.12:
        return (
            "pear",
            "The hip line reads wider than the shoulders. Structure the top (collars, jackets) "
            "and keep bottoms in a continuous dark or mid tone.",
        )
    if waist and shoulders and hips and waist < min(shoulders, hips) * 0.86:
        return (
            "hourglass",
            "Waist is narrower than shoulder and hip. Belted jackets, defined waists, "
            "and dresses that skim rather than swamp the middle will read best.",
        )
    if shoulders and hips and abs(shoulders - hips) < 0.08:
        return (
            "rectangle",
            "Shoulder and hip widths are close. Create shape with layering, a belt, "
            "or a cropped jacket over a longer line.",
        )
    return (
        "athletic",
        "Proportions look balanced and straight. Tailored layers and a clear waist or hem "
        "line will add interest without fighting the frame.",
    )


def fallback_analysis(image: Image.Image) -> dict:
    """SegFormer geometry + face color when Gemini is unavailable."""
    try:
        load_segformer()
        _, _, label_map = segment_clothing(image, category="upper")
        arr = np.asarray(image.convert("RGB"))
        if arr.shape[:2] != label_map.shape:
            label_map = np.array(
                Image.fromarray(label_map.astype(np.uint8)).resize(image.size, Image.Resampling.NEAREST)
            )
        face = arr[label_map == 11]
        rgb = _mean_rgb(face)
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
            f"Lean into {season} colors and a {body_type} silhouette: "
            "one tailored layer, one easy piece, and shoes that stay in the same value family."
        ),
        "silhouette_tips": body_notes,
        "occasions": ["weekday", "smart casual"],
        "used_gemini": False,
    }


def analyze_avatar(image: Image.Image) -> dict:
    payload, used = analyze_avatar_llm(image)
    if used and payload.get("color_season") and payload.get("body_type"):
        payload["used_gemini"] = True
        payload.setdefault("palette", [])
        payload.setdefault("avoid_colors", [])
        payload.setdefault("occasions", [])
        return payload
    data = fallback_analysis(image)
    if payload:
        data.update({k: v for k, v in payload.items() if v not in (None, "", [], {})})
        data["used_gemini"] = used
    return data


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
