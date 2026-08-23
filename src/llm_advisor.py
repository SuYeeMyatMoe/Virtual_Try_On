"""Gemini fashion advisor: garment caption, styling advice, result explanation.

Uses GOOGLE_API_KEY when present. Falls back to template copy otherwise.
"""

from __future__ import annotations

import io
import os
from typing import Any, Optional, Sequence

from pathlib import Path

from dotenv import load_dotenv
from PIL import Image

from .preprocess import normalize_garment_region

_ENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(_ENV, override=True)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GRANITE_MODEL = os.getenv("GRANITE_MODEL", "ibm-granite/granite-4.1-8b")
GRANITE_URL = os.getenv(
    "GRANITE_INFERENCE_URL",
    "https://router.huggingface.co/v1/chat/completions",
)
CAPTION_TIMEOUT = 25
GRANITE_SYSTEM = (
    "You write short fashion shopping prompts. Return JSON only, no markdown. "
    "Keys: prompt (one sentence about the garment only), "
    "dress_query (2-6 words for catalog search), "
    "garment_category (dress, upper, or lower). "
    "Do not mention shoes, heels, or footwear. Do not invent brand names."
)

SYSTEM_STYLIST = (
    "You are VESTURE, a world-class luxury fashion stylist with exceptional high-fashion sense "
    "and elite recommendation judgment. "
    "Write 2–4 short sentences. No hashtags. No emoji. "
    "Ground every claim in the garment description, confidence scores, "
    "or recommended catalog titles you are given. Do not invent brands."
)

SYSTEM_CHAT = (
    "You are VESTURE, a friendly personal shopper. Talk like a helpful person, not a runway critic. "
    "Use everyday words: shirt, trousers, jacket, dress — not 'column', 'editorial', 'atelier', or 'silhouette' unless needed. "
    "Reply in markdown with short lines:\n"
    "1) One sentence: what to wear, in plain language.\n"
    "2) One sentence: why it suits their colors or body.\n"
    "3) If catalog pieces are listed, a numbered list of up to 3, each on its own line: "
    "'1. **Title** — short reason'. Do not dump every title into one sentence.\n"
    "Keep the whole reply under 90 words. Blank line between the plan and the list. "
    "If presentation is man/menswear, only men's or unisex pieces — never a women's dress, kurta, skirt, or blouse. "
    "If presentation is woman/womenswear, skip men's-only pieces. "
    "Do not open with 'Menswear,' or the season name alone. "
    "Never tell them to upload an avatar or tap Analyze if a profile is already present. "
    "No hashtags. No emoji.\n"
    "Example — shopper: What colors suit me? Profile: Soft Summer, dusty rose, sage, taupe, navy.\n"
    "You: Your coloring is a cool Soft Summer.\n\n"
    "**Good on you:** dusty rose, sage, taupe, and soft navy.\n\n"
    "**Skip:** hot pink, orange, and stark white.\n"
    "Example — shopper: I want to wear for a fashion show. Profile: inverted triangle, Soft Summer, man.\n"
    "You: For a show, wear an easy open shirt with straight navy trousers — nothing boxy on the shoulders.\n\n"
    "Those Soft Summer colors stay quiet on your skin.\n\n"
    "1. **John Players Men Navy Blue Shirt** — open neck, easy shoulder\n"
    "2. **Peter England Men Party Blue Jeans** — clean straight line"
)

FEW_SHOT_CAPTION = (
    "Example: a navy cotton crew-neck t-shirt with a relaxed fit and matte finish.\n"
    "Example: a black tailored blazer in structured wool with notch lapels.\n"
    "Example: dark-wash slim jeans with a mid rise and visible stitching.\n"
    "Example: a camel midi skirt in smooth wool with a clean A-line."
)


def has_gemini() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def _api_key() -> Optional[str]:
    return os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or None


def _image_bytes(image: Image.Image) -> bytes:
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _generate_text(
    prompt: str,
    image: Optional[Image.Image] = None,
    *,
    max_output_tokens: int = 280,
    temperature: float = 0.4,
    system_instruction: str = SYSTEM_STYLIST,
    json_mode: bool = False,
) -> str:
    key = _api_key()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        parts: list[Any] = [prompt]
        if image is not None:
            parts = [
                types.Part.from_bytes(data=_image_bytes(image), mime_type="image/jpeg"),
                prompt,
            ]
        cfg_kw: dict[str, Any] = {
            "system_instruction": system_instruction,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
        }
        if json_mode:
            cfg_kw["response_mime_type"] = "application/json"
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=parts,
            config=types.GenerateContentConfig(**cfg_kw),
        )
        text = (getattr(response, "text", None) or "").strip()
        if text:
            return text
    except Exception:
        pass

    import google.generativeai as genai_old

    genai_old.configure(api_key=key)
    model = genai_old.GenerativeModel(
        DEFAULT_MODEL if DEFAULT_MODEL.startswith("gemini") else "gemini-1.5-flash",
        system_instruction=system_instruction,
    )
    contents: list[Any] = [prompt]
    if image is not None:
        contents = [image.convert("RGB"), prompt]
    response = model.generate_content(contents)
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def _chat_turns(history: Optional[Sequence[dict]], user_message: str) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    for msg in list(history or [])[-8:]:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        role = "user" if msg.get("role") == "user" else "model"
        turns.append({"role": role, "text": content})
    user_message = (user_message or "").strip()
    if user_message and (not turns or turns[-1]["role"] != "user" or turns[-1]["text"] != user_message):
        turns.append({"role": "user", "text": user_message})
    if not turns:
        turns.append({"role": "user", "text": user_message or "Hello"})
    # Gemini requires the first turn to be from the user
    if turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "text": "You are my stylist. Let's talk fashion."})
    return turns


def _user_parts(
    text: str,
    images: Optional[Sequence[Image.Image]] = None,
    audio_bytes: Optional[bytes] = None,
    audio_mime: str = "audio/wav",
) -> list[Any]:
    from google.genai import types

    parts: list[Any] = []
    for img in list(images or []):
        parts.append(types.Part.from_bytes(data=_image_bytes(img), mime_type="image/jpeg"))
    if audio_bytes:
        parts.append(types.Part.from_bytes(data=audio_bytes, mime_type=audio_mime or "audio/wav"))
    parts.append(types.Part.from_text(text=text))
    return parts


def _generate_chat(
    history: Optional[Sequence[dict]],
    user_message: str,
    *,
    context: str = "",
    images: Optional[Sequence[Image.Image]] = None,
    audio_bytes: Optional[bytes] = None,
    audio_mime: str = "audio/wav",
    max_output_tokens: int = 512,
    temperature: float = 0.6,
    system_instruction: str = SYSTEM_CHAT,
) -> str:
    key = _api_key()
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    turns = _chat_turns(history, user_message)
    if context and turns and turns[-1]["role"] == "user":
        turns[-1] = {
            "role": "user",
            "text": f"{context.rstrip()}\n\nShopper: {turns[-1]['text']}",
        }

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        contents = [
            types.Content(role=t["role"], parts=[types.Part.from_text(text=t["text"])])
            for t in turns[:-1]
        ]
        contents.append(
            types.Content(
                role="user",
                parts=_user_parts(turns[-1]["text"], images, audio_bytes, audio_mime),
            )
        )
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        text = (getattr(response, "text", None) or "").strip()
        if text:
            return text
    except Exception:
        pass

    import google.generativeai as genai_old

    genai_old.configure(api_key=key)
    model = genai_old.GenerativeModel(
        DEFAULT_MODEL if DEFAULT_MODEL.startswith("gemini") else "gemini-1.5-flash",
        system_instruction=system_instruction,
    )
    hist = []
    for t in turns[:-1]:
        hist.append({"role": "user" if t["role"] == "user" else "model", "parts": [t["text"]]})
    chat = model.start_chat(history=hist)
    last_parts: list[Any] = [img.convert("RGB") for img in list(images or [])]
    last_parts.append(turns[-1]["text"])
    response = chat.send_message(last_parts)
    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response.")
    return text


def _template_caption(
    category: str = "upper",
    color: Optional[str] = None,
    style: Optional[str] = None,
) -> str:
    kind = {
        "upper": "top or shirt",
        "lower": "pair of pants or a skirt",
        "dress": "dress",
    }.get(normalize_garment_region(category), "fashion garment")
    bits = [p for p in (color, style) if p]
    look = " ".join(bits + [kind]) if bits else kind
    return f"a {look} with clean lines and wearable everyday styling"


def caption_garment(
    image: Optional[Image.Image],
    category: str = "upper",
    color: Optional[str] = None,
    style: Optional[str] = None,
) -> tuple[str, bool]:
    """
    Describe the garment for IDM-VTON `garment_des` / SD prompt.

    Returns (caption, used_gemini).
    """
    fallback = _template_caption(category, color, style)
    if image is None or not has_gemini():
        return fallback, False
    region = normalize_garment_region(category)
    region_lock = {
        "lower": (
            "This is a LOWER-BODY garment (pants, jeans, shorts, or skirt). "
            "Never describe it as a shirt, top, blouse, or jacket."
        ),
        "dress": "This is a DRESS. Do not describe it as a separate top.",
        "upper": "This is an UPPER-BODY garment (shirt, blouse, jacket, or knit).",
    }.get(region, "")
    prompt = (
        f"{FEW_SHOT_CAPTION}\n"
        "Describe this garment in one sentence for a virtual try-on model. "
        "Include color, fabric, silhouette, and garment type. "
        f"Category hint: {region}. {region_lock}"
    )
    try:
        text = _generate_text(prompt, image=image)
        return text.split("\n")[0].strip().strip('"'), True
    except Exception:
        return fallback, False


def _top5_line(top5: Optional[Sequence[dict]]) -> str:
    if not top5:
        return "No catalog matches yet."
    parts = []
    for item in list(top5)[:5]:
        title = item.get("title", "item")
        sim = item.get("similarity")
        color = item.get("color")
        label = str(title)
        if color:
            label = f"{title} ({color})"
        if isinstance(sim, (int, float)):
            parts.append(f"{label} {sim:.0%}")
        else:
            parts.append(label)
    return "; ".join(parts)


def _top5_chat_list(top5: Optional[Sequence[dict]], limit: int = 3) -> str:
    if not top5:
        return ""
    lines = []
    for i, item in enumerate(list(top5)[:limit], 1):
        title = str(item.get("title") or "item")
        color = str(item.get("color") or "").strip()
        extra = f" — {color}" if color else ""
        lines.append(f"{i}. **{title}**{extra}")
    return "\n".join(lines)


def _soften_chat_text(text: str) -> str:
    """Turn a one-block reply into short paragraphs the shopper can scan."""
    import re

    t = " ".join(str(text or "").split())
    if not t:
        return t
    if "\n" in str(text or "") and len(str(text).strip().splitlines()) >= 2:
        return str(text).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    if len(sentences) <= 2:
        return t
    chunks: list[str] = []
    buf: list[str] = []
    for sent in sentences:
        buf.append(sent)
        if len(buf) >= 2:
            chunks.append(" ".join(buf))
            buf = []
    if buf:
        chunks.append(" ".join(buf))
    return "\n\n".join(chunks)


def _has_profile(analysis: Optional[dict]) -> bool:
    analysis = analysis or {}
    return bool(analysis.get("color_season") or analysis.get("body_type") or analysis.get("palette"))


def _palette_line(analysis: dict) -> str:
    colors = [str(c) for c in (analysis.get("palette") or []) if str(c).strip()]
    return ", ".join(colors[:6])


def _avoid_line(analysis: dict) -> str:
    colors = [str(c) for c in (analysis.get("avoid_colors") or []) if str(c).strip()]
    return ", ".join(colors[:4])


def _body_line(body_type: str, presentation: str = "") -> str:
    key = str(body_type or "").strip().lower()
    menswear = str(presentation or "").strip().lower() in {"man", "male", "men", "menswear"}
    if menswear:
        tips = {
            "inverted triangle": (
                "Ease the shoulder with an open shirt or unconstructed jacket, "
                "and keep trousers a clean column."
            ),
            "pear": "Give the top some structure — a collared shirt or light layer — and keep the lower half simple.",
            "apple": "A longer shirt or light overshirt through the middle keeps proportion calm.",
            "hourglass": "A jacket that nips in or a tucked shirt will follow the frame.",
            "rectangle": "A break at the waist — tuck, knit, or cropped layer — adds shape.",
            "athletic": "Tailored layers and a clear trouser hem add interest to a straight frame.",
        }
        return tips.get(key, "Keep proportion clean: one tailored piece, one easy piece.")
    tips = {
        "inverted triangle": (
            "Ease the shoulder — a fluid blouse or open jacket, not a boxy blazer — "
            "and give the lower half a strong column."
        ),
        "pear": "Structure the top and keep the lower half in a continuous mid or dark tone.",
        "apple": "A longer line through the middle keeps proportion calm; avoid cling at the waist.",
        "hourglass": "A defined waist or a jacket that nips in will follow the frame.",
        "rectangle": "A break at the waist — belt, tuck, or cropped layer — adds shape.",
        "athletic": "Tailored layers and a clear hem line add interest to a straight frame.",
    }
    return tips.get(key, "Keep proportion clean: one tailored piece, one easy piece.")


def local_stylist_reply(
    user_message: str,
    analysis: Optional[dict] = None,
    recs: Optional[Sequence[dict]] = None,
) -> str:
    """Grounded reply when Gemini is down. Uses the analyzed profile; never asks to re-analyze."""
    analysis = analysis or {}
    text = (user_message or "").lower()
    season = str(analysis.get("color_season") or "").strip()
    undertone = str(analysis.get("undertone") or "").strip()
    body = str(analysis.get("body_type") or "").strip()
    presentation = str(analysis.get("presentation") or "").strip().lower()
    dept = (
        "menswear"
        if presentation in {"man", "male", "men", "menswear"}
        else "womenswear"
        if presentation in {"woman", "female", "women", "womenswear"}
        else ""
    )
    palette = _palette_line(analysis)
    avoid = _avoid_line(analysis)

    if not _has_profile(analysis):
        return (
            "I can still help in simple terms.\n\n"
            "**Wear:** one clear shape and a small palette — navy, taupe, or ivory.\n\n"
            "Add a photo and tap Analyze when you want colors matched to your skin."
        )

    color_q = any(
        p in text
        for p in ("color", "colour", "palette", "suit me", "undertone", "season", "wearable")
    )
    show_q = any(
        p in text for p in ("fashion show", "runway", "editorial", "walk", "couture")
    )
    weekend_q = any(p in text for p in ("weekend", "casual", "everyday"))
    formal_q = any(p in text for p in ("wedding", "gala", "interview", "office", "party", "date"))
    rec_list = _top5_chat_list(recs)

    if color_q and not show_q:
        bits = [
            f"Your coloring reads **{undertone or 'balanced'}**"
            + (f" in the **{season}** family" if season else "")
            + "."
        ]
        if palette:
            bits.append(f"**Good on you:** {palette}.")
        if avoid:
            bits.append(f"**Skip:** {avoid}.")
        if body:
            bits.append(_body_line(body, presentation))
        return "\n\n".join(bits)

    if show_q:
        if dept == "menswear":
            look = (
                "For a show, wear an easy open shirt with straight trousers — nothing boxy on the shoulders.\n\n"
                f"Stay in {season or 'your'} colors"
                + (f" ({palette})" if palette else "")
                + ".\n\n"
                + _body_line(body, presentation)
            )
        else:
            look = (
                "For a show, keep the top easy and the lower half a simple matching line.\n\n"
                f"Stay in {season or 'your'} colors"
                + (f" ({palette})" if palette else "")
                + ".\n\n"
                + _body_line(body, presentation)
            )
    elif weekend_q:
        look = (
            f"**Weekend:** keep it easy in {palette or 'muted neutrals'} — "
            "a simple top and a matching lower half.\n\n"
            + _body_line(body, presentation)
        )
    elif formal_q:
        look = (
            f"**For this occasion:** hold {season or 'your'} colors and a calm shape.\n\n"
            + _body_line(body, presentation)
            + (f"\n\nLean on {palette}." if palette else "")
        )
    else:
        look = (
            f"Try {palette or season or 'your colors'} on a {body or 'balanced'} frame"
            + (f" ({dept})" if dept else "")
            + ".\n\n"
            + _body_line(body, presentation)
        )

    if rec_list:
        look += "\n\n**From the catalog:**\n" + rec_list
    return look


def style_advice(
    garment_desc: str,
    category: str = "upper",
    top5: Optional[Sequence[dict]] = None,
    scores: Optional[dict] = None,
) -> tuple[str, bool]:
    """Occasion / pairing advice. Returns (text, used_gemini)."""
    scores = scores or {}
    top_line = _top5_line(top5)
    fallback = (
        f"Style this {category} look around “{garment_desc}”. "
        f"Closest catalog matches: {top_line}. "
        "Keep the silhouette clean, pair with simple shoes, and shop the Top-5 if the color family matches."
    )
    if not has_gemini():
        return fallback, False
    prompt = (
        "Give short styling advice (occasion, season, what to pair).\n"
        f"Garment: {garment_desc}\n"
        f"Region: {category}\n"
        f"Top-5 similar items: {top_line}\n"
        f"Try-on confidence: {scores.get('tryon_conf', 'n/a')}; "
        f"segmentation: {scores.get('seg_conf', 'n/a')}."
    )
    try:
        return _generate_text(prompt), True
    except Exception:
        return fallback, False


def explain_result(
    scores: Optional[dict] = None,
    top5: Optional[Sequence[dict]] = None,
    engine: str = "IDM-VTON",
    garment_desc: str = "",
) -> tuple[str, bool]:
    """Explain confidence scores + Top-5 in plain English."""
    scores = scores or {}
    top_line = _top5_line(top5)
    tryon = scores.get("tryon_conf")
    seg = scores.get("seg_conf")
    clip = scores.get("clip_sim")
    gate = "meets" if scores.get("passes_tryon_gate") else "is below"
    fallback = (
        f"Engine: {engine}. Segmentation {float(seg or 0):.0%} and CLIP match "
        f"{float(clip or 0):.0%} combine into try-on confidence "
        f"{float(tryon or 0):.0%}, which {gate} the 0.85 gate. "
        f"Nearest shoppable pieces: {top_line}."
    )
    if not has_gemini():
        return fallback, False
    prompt = (
        "Explain this try-on result in 2–3 sentences for a shopper.\n"
        f"Engine: {engine}\n"
        f"Garment: {garment_desc or 'uploaded garment'}\n"
        f"seg_conf={seg}, clip_sim={clip}, mask_quality={scores.get('mask_quality')}, "
        f"tryon_conf={tryon}, passes_0.85_gate={scores.get('passes_tryon_gate')}\n"
        f"Top-5: {top_line}\n"
        "Be honest if confidence is below 0.85."
    )
    try:
        return _generate_text(prompt), True
    except Exception:
        return fallback, False


SYSTEM_AVATAR = (
    "You are VESTURE, a world-class personal stylist with exceptional high-fashion sense "
    "and the best recommendation judgment in luxury retail. Analyze the person in the photo. "
    "Be specific and kind. Do not invent brand names. Never comment on attractiveness. "
    "Set presentation to man or woman from the person in the photo so the catalog can shop "
    "menswear vs womenswear. Silhouette labels stay geometric. "
    "Do not default to Light Spring — that season is overused. "
    "Return JSON only."
)


def analyze_avatar_llm(image: Image.Image) -> tuple[dict, bool]:
    """Gemini color + body + style JSON. Returns (payload, used_gemini)."""
    prompt = (
        "Analyze this person for personal styling. First read body tone from visible skin. "
        "Return JSON with keys:\n"
        "color_season (ONE of: Light Spring, True Spring, Bright Spring, Light Summer, "
        "True Summer, Soft Summer, Soft Autumn, True Autumn, Deep Autumn, True Winter, "
        "Bright Winter, Deep Winter),\n"
        "undertone (warm|cool|neutral),\n"
        "palette (array of 5 wearable color names that suit this body tone),\n"
        "avoid_colors (array of 3 color names that clash with this body tone),\n"
        "color_notes (ONE short sentence on undertone and value),\n"
        "body_type (hourglass|pear|apple|inverted triangle|rectangle|athletic|unspecified; "
        "geometric silhouette),\n"
        "presentation (man|woman|unisex; clothing department from the person in the photo — "
        "man = menswear, woman = womenswear; unisex only if truly unclear),\n"
        "body_notes (ONE short sentence on shoulder / waist / hip only; unspecified if not full body),\n"
        "style_direction (ONE short sentence of what to wear in that department; do not repeat body_notes),\n"
        "silhouette_tips (empty string, or one different short tip — never copy body_notes),\n"
        "occasions (array of 2 occasions).\n"
        "Do not default to Light Spring. Olive, golden-medium, or muted warm skin is usually "
        "Soft Autumn or True Autumn. Cool pink skin is Summer or Winter. "
        "Light Spring only if the coloring is clearly warm, light, and high-chroma.\n"
        "If the photo is a crop or unclear, say so honestly."
    )
    if not has_gemini():
        return {}, False
    try:
        text = _generate_text(
            prompt,
            image=image,
            max_output_tokens=700,
            temperature=0.3,
            system_instruction=SYSTEM_AVATAR,
            json_mode=True,
        )
        return _parse_json_object(text), True
    except Exception:
        try:
            text = _generate_text(
                prompt,
                image=image,
                max_output_tokens=700,
                temperature=0.3,
                system_instruction=SYSTEM_AVATAR,
            )
            return _parse_json_object(text), True
        except Exception:
            return {}, False


def _audio_bytes(audio: Any) -> bytes:
    if audio is None:
        return b""
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)
    if hasattr(audio, "getvalue"):
        data = audio.getvalue()
        if data:
            return bytes(data)
    if hasattr(audio, "read"):
        try:
            audio.seek(0)
        except Exception:
            pass
        data = audio.read()
        if data:
            return bytes(data)
    return b""


def _sniff_audio_mime(data: bytes, claimed: str = "") -> str:
    claimed = str(claimed or "").split(";")[0].strip().lower()
    if data.startswith(b"RIFF") and b"WAVE" in data[:16]:
        return "audio/wav"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    if data.startswith(b"ID3") or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if data[:4] == b"\x1aE\xdf\xa3":
        return "audio/webm"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "audio/mp4"
    if claimed.startswith("audio/"):
        return claimed
    return "audio/wav"


def _gemini_response_text(response: Any) -> str:
    text = (getattr(response, "text", None) or "").strip().strip('"').strip()
    if text:
        return text
    for cand in getattr(response, "candidates", None) or []:
        content = getattr(cand, "content", None)
        for part in getattr(content, "parts", None) or []:
            piece = (getattr(part, "text", None) or "").strip().strip('"').strip()
            if piece:
                return piece
    return ""


def transcribe_audio(audio: Any, mime_type: str = "audio/wav") -> tuple[str, str]:
    """Transcribe a recorded voice note. Returns (transcript, error)."""
    load_dotenv(_ENV, override=True)
    key = _api_key()
    if not key:
        return "", "Voice needs a Gemini key. Add GOOGLE_API_KEY to .env, then restart Streamlit."
    data = _audio_bytes(audio)
    if len(data) < 2500:
        return "", "That recording was too short. Hold the mic, speak a full sentence, then send."
    claimed = getattr(audio, "type", None) or mime_type or "audio/wav"
    mime = _sniff_audio_mime(data, str(claimed))
    prompt = (
        "Transcribe the spoken words in this audio exactly. "
        "Return only the transcript, no quotes, no extra commentary."
    )
    mimes = []
    for candidate in (mime, "audio/wav", "audio/webm", "audio/ogg", "audio/mp4", "audio/mpeg"):
        if candidate not in mimes:
            mimes.append(candidate)
    models = []
    for name in (DEFAULT_MODEL, "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"):
        if name and name not in models:
            models.append(name)

    last_err = "Gemini returned no transcript."
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        for model_name in models[:2]:
            for try_mime in mimes[:3]:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=[
                            types.Part.from_bytes(data=data, mime_type=try_mime),
                            prompt,
                        ],
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            max_output_tokens=400,
                        ),
                    )
                    text = _gemini_response_text(response)
                    if text and len(text) > 1:
                        return text, ""
                    finish = ""
                    cands = getattr(response, "candidates", None) or []
                    if cands:
                        finish = str(getattr(cands[0], "finish_reason", "") or "")
                    if finish:
                        last_err = f"Gemini did not return words ({finish})."
                except Exception as exc:
                    last_err = str(exc)
                    continue
    except Exception as exc:
        last_err = str(exc)

    try:
        import google.generativeai as genai_old

        genai_old.configure(api_key=key)
        model = genai_old.GenerativeModel(models[0])
        response = model.generate_content(
            [{"mime_type": mime, "data": data}, prompt]
        )
        text = _gemini_response_text(response)
        if text and len(text) > 1:
            return text, ""
    except Exception as exc:
        last_err = str(exc)

    short = (last_err or "Could not transcribe that clip.").replace("\n", " ").strip()
    return "", short[:280]


def stylist_chat(
    user_message: str,
    history: Optional[Sequence[dict]] = None,
    analysis: Optional[dict] = None,
    recs: Optional[Sequence[dict]] = None,
    images: Optional[Sequence[Image.Image]] = None,
    audio_bytes: Optional[bytes] = None,
    audio_mime: str = "audio/wav",
) -> tuple[str, bool]:
    """Conversational styling reply via Gemini. Returns (text, used_gemini)."""
    analysis = analysis or {}
    rec_line = _top5_line(recs)
    rec_list = _top5_chat_list(recs)
    palette = ", ".join(str(c) for c in (analysis.get("palette") or [])[:6])
    context_bits = []
    if analysis.get("color_season") or analysis.get("body_type"):
        context_bits.append(
            "Client profile: "
            f"color season={analysis.get('color_season') or 'n/a'}, "
            f"undertone={analysis.get('undertone') or 'n/a'}, "
            f"palette={palette or 'n/a'}, "
            f"body type={analysis.get('body_type') or 'n/a'}, "
            f"presentation={analysis.get('presentation') or 'n/a'}."
        )
        if analysis.get("body_notes"):
            context_bits.append(str(analysis["body_notes"]))
        if analysis.get("style_direction"):
            context_bits.append(str(analysis["style_direction"]))
    color_q = any(
        p in (user_message or "").lower()
        for p in ("what color", "which color", "suit me", "my palette", "undertone")
    )
    if rec_list and not color_q:
        context_bits.append(
            "Catalog pieces on the table (pick up to 3; skip any that clash with the palette):\n"
            f"{rec_list}"
        )
    elif rec_line and rec_line != "No catalog matches yet." and not color_q:
        context_bits.append(
            "Catalog pieces on the table (prefer palette matches; reject pieces that clash): "
            f"{rec_line}."
        )
    if images:
        context_bits.append(
            f"The shopper attached {len(list(images))} look/garment photo(s). "
            "Read clothing, color, silhouette, and whether this is a man or woman "
            "before recommending. Stay in the matching department."
        )
    if audio_bytes:
        context_bits.append("The shopper also sent a voice note; honor that brief.")
    context = "\n".join(context_bits)
    fallback = local_stylist_reply(user_message, analysis, recs)
    packed = (
        f"{context}\n\nShopper: {user_message}\n\n"
        "Reply as VESTURE. If a profile is present, answer from it. "
        "Do not ask them to analyze again."
    )
    if not has_gemini():
        return fallback, False
    prior = [m for m in list(history or [])[:-1] if isinstance(m, dict)]
    try:
        text = _generate_chat(
            prior,
            user_message,
            context=context,
            images=images,
            audio_bytes=audio_bytes,
            audio_mime=audio_mime,
            max_output_tokens=512,
            temperature=0.6,
            system_instruction=SYSTEM_CHAT,
        )
        if text and "tap Analyze" not in text and "Upload an avatar" not in text:
            return _soften_chat_text(text), True
    except Exception:
        pass
    try:
        text = _generate_text(
            packed,
            max_output_tokens=512,
            temperature=0.55,
            system_instruction=SYSTEM_CHAT,
        )
        if text and "tap Analyze" not in text and "Upload an avatar" not in text:
            return _soften_chat_text(text), True
    except Exception:
        pass
    return fallback, False


def _parse_json_object(text: str) -> dict:
    import json
    import re

    blob = (text or "").strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:json)?\s*", "", blob)
        blob = re.sub(r"\s*```$", "", blob)
    match = re.search(r"\{.*\}", blob, flags=re.DOTALL)
    if match:
        blob = match.group(0)
    data = json.loads(blob)
    return data if isinstance(data, dict) else {}


_COLOR_WORDS = (
    "black", "white", "navy", "red", "blue", "green", "pink", "beige", "grey",
    "gray", "brown", "olive", "cream", "ivory", "yellow", "purple", "gold",
    "silver", "coral", "burgundy", "khaki", "charcoal", "nude",
)


def _fallback_look_prompt(user_text: str, analysis: Optional[dict] = None) -> dict:
    """Local prompt if Granite is unreachable. Garment only — no shoes."""
    analysis = analysis or {}
    t = (user_text or "").lower()
    color = next((c for c in _COLOR_WORDS if c in t), "")
    if not color:
        palette = analysis.get("palette") or []
        color = str(palette[0]).split()[0].lower() if palette else "black"
    if color == "gray":
        color = "grey"
    if any(k in t for k in ("jean", "pant", "trouser", "chino")):
        garment = f"{color} trousers"
        category = "lower"
    elif any(k in t for k in ("shirt", "tee", "blouse", "jacket", "hoodie", "blazer")):
        garment = f"{color} shirt"
        category = "upper"
    else:
        garment = f"{color} dress"
        category = "dress"
    prompt = f"{garment}, clean silhouette"
    return {
        "prompt": prompt,
        "dress_query": garment,
        "garment_category": category,
        "source": "fallback",
    }


def _normalize_look_payload(data: dict, user_text: str, analysis: Optional[dict]) -> dict:
    fallback = _fallback_look_prompt(user_text, analysis)
    prompt = str(data.get("prompt") or fallback["prompt"]).strip()
    dress_query = str(data.get("dress_query") or fallback["dress_query"]).strip()
    category = str(data.get("garment_category") or fallback["garment_category"]).strip().lower()
    if category not in {"dress", "upper", "lower"}:
        category = fallback["garment_category"]
    if not prompt or not dress_query:
        return fallback
    return {
        "prompt": prompt,
        "dress_query": dress_query,
        "garment_category": category,
        "source": str(data.get("source") or "granite"),
    }


def granite_look_prompt(user_text: str, analysis: Optional[dict] = None) -> dict:
    """IBM Granite 4.1-8B writes a try-on / shopping prompt. Falls back locally."""
    analysis = analysis or {}
    fallback = _fallback_look_prompt(user_text, analysis)
    from .hf_auth import ensure_hf_login, hf_token

    ensure_hf_login()
    token = hf_token()
    if not token:
        return fallback

    palette = ", ".join(str(c) for c in (analysis.get("palette") or [])[:5])
    user = (
        f"Shopper request: {user_text}\n"
        f"Color season: {analysis.get('color_season') or 'n/a'}. "
        f"Palette: {palette or 'n/a'}. "
        f"Department: {analysis.get('presentation') or 'n/a'}.\n"
        "Write JSON for a matching garment only. Do not mention shoes."
    )
    try:
        import requests

        resp = requests.post(
            GRANITE_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "model": GRANITE_MODEL,
                "messages": [
                    {"role": "system", "content": GRANITE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "max_tokens": 180,
                "temperature": 0.3,
            },
            timeout=45,
        )
        resp.raise_for_status()
        payload = resp.json()
        text = (
            (((payload.get("choices") or [{}])[0].get("message") or {}).get("content"))
            or payload.get("generated_text")
            or ""
        )
        data = _parse_json_object(str(text))
        if data:
            return _normalize_look_payload(data, user_text, analysis)
    except Exception as exc:
        print(f"Granite look prompt failed ({exc}). Using local fallback.")
    return fallback

