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
CAPTION_TIMEOUT = 25

SYSTEM_STYLIST = (
    "You are VESTURE, a world-class luxury fashion stylist with exceptional high-fashion sense "
    "and elite recommendation judgment. "
    "Write 2–4 short sentences. No hashtags. No emoji. "
    "Ground every claim in the garment description, confidence scores, "
    "or recommended catalog titles you are given. Do not invent brands."
)

SYSTEM_CHAT = (
    "You are VESTURE, a world-class personal stylist with exceptional high-fashion sense "
    "and the sharpest outfit-recommendation judgment in luxury retail. "
    "Think like a creative director: silhouette, color theory, fabric, proportion, occasion, "
    "and what will actually flatter this client. Recommend with confidence — what to wear, "
    "why it works, and what to pair. When catalog pieces are listed, rank them and say which "
    "to try first. Prefer pieces that sit in the client's palette; say so if a listed piece "
    "is too harsh (for example stark black on a Soft Summer). "
    "When the shopper shares a photo, read the garment, color, and fit before advising. "
    "When they send a voice note, treat the transcript as their brief. "
    "If a client profile exists, stay inside that palette and body type. "
    "Never tell them to upload an avatar or tap Analyze if a color season or body type is already in the profile. "
    "If nothing has been analyzed yet, still give precise general advice. "
    "Reply in 2–6 short sentences. Do not invent brand names. No hashtags. No emoji.\n"
    "Example — shopper: What colors suit me? Profile: Soft Summer, dusty rose, sage, taupe, navy.\n"
    "You: Your Soft Summer set is dusty rose, sage, taupe, soft navy, and mauve. "
    "Those muted cool-neutrals sit quietly on your skin; ease off hot pink, orange, and stark white.\n"
    "Example — shopper: I want to wear for a fashion show. Profile: inverted triangle, Soft Summer.\n"
    "You: Keep the shoulder line soft — a fluid dusty-rose or taupe blouse, not a boxy black blazer. "
    "Give the lower half a clean navy or mauve column so the walk reads editorial. "
    "One metallic or leather accessory is enough; skip a tracksuit."
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


def _has_profile(analysis: Optional[dict]) -> bool:
    analysis = analysis or {}
    return bool(analysis.get("color_season") or analysis.get("body_type") or analysis.get("palette"))


def _palette_line(analysis: dict) -> str:
    colors = [str(c) for c in (analysis.get("palette") or []) if str(c).strip()]
    return ", ".join(colors[:6])


def _avoid_line(analysis: dict) -> str:
    colors = [str(c) for c in (analysis.get("avoid_colors") or []) if str(c).strip()]
    return ", ".join(colors[:4])


def _body_line(body_type: str) -> str:
    key = str(body_type or "").strip().lower()
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
    palette = _palette_line(analysis)
    avoid = _avoid_line(analysis)
    rec_line = _top5_line(recs)
    has_recs = rec_line != "No catalog matches yet."

    if not _has_profile(analysis):
        return (
            "I can still style you in general terms. For a fashion show, pick one clear silhouette "
            "and a tight palette — navy, taupe, or ivory — and keep the walk uncluttered. "
            "Upload an avatar and tap Analyze if you want colors locked to your skin."
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

    if color_q and not show_q:
        bits = [
            f"Your reading is {undertone or 'balanced'} undertone"
            + (f" in the {season} family" if season else "")
            + "."
        ]
        if palette:
            bits.append(f"Wear {palette} — those sit quietly on your skin.")
        if avoid:
            bits.append(f"Ease off {avoid}.")
        if body:
            bits.append(_body_line(body))
        return " ".join(bits)

    occasion = "a fashion-show walk" if show_q else (
        "a casual weekend" if weekend_q else (
            "a more formal moment" if formal_q else "this occasion"
        )
    )
    if show_q:
        look = (
            f"For {occasion}, stay inside {season or 'your'} colors"
            + (f" ({palette})" if palette else "")
            + ". "
            + _body_line(body)
            + " One editorial idea: a fluid dusty-rose or taupe top with a navy or mauve column, "
            "clean shoes, one metal or leather accent. Skip a tracksuit and skip a heavy black blazer "
            "if it fights a soft palette."
        )
    elif weekend_q:
        look = (
            f"For {occasion}, keep {season or 'your palette'} easy: "
            f"{palette or 'muted neutrals'} in a simple top and a continuous lower half. "
            + _body_line(body)
        )
    elif formal_q:
        look = (
            f"For {occasion}, hold the {season or 'personal'} palette and a calm silhouette. "
            + _body_line(body)
            + (f" Lean on {palette}." if palette else "")
        )
    else:
        look = (
            f"{season or 'Your palette'}"
            + (f" ({palette})" if palette else "")
            + f" with a {body or 'balanced'} frame. "
            + _body_line(body)
        )

    if has_recs:
        look += f" Closest catalog pieces in this edit: {rec_line}."
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
    "Do not assign gender. Silhouette labels are geometric and apply to everyone. "
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
        "geometric silhouette for any gender),\n"
        "body_notes (ONE short sentence on shoulder / waist / hip only; unspecified if not full body; "
        "never assign gender),\n"
        "style_direction (ONE short sentence of what to wear; do not repeat body_notes),\n"
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


def transcribe_audio(audio: Any, mime_type: str = "audio/wav") -> tuple[str, bool]:
    """Transcribe a recorded voice note. Returns (transcript, used_gemini)."""
    if audio is None or not has_gemini():
        return "", False
    data = audio.getvalue() if hasattr(audio, "getvalue") else bytes(audio)
    if not data:
        return "", False
    mime = getattr(audio, "type", None) or mime_type or "audio/wav"
    prompt = "Transcribe this spoken fashion request exactly. Return only the transcript, no quotes."
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=_api_key())
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=[types.Part.from_bytes(data=data, mime_type=mime), prompt],
        )
        text = (getattr(response, "text", None) or "").strip().strip('"')
        return text, bool(text)
    except Exception:
        return "", False


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
    palette = ", ".join(str(c) for c in (analysis.get("palette") or [])[:6])
    context_bits = []
    if analysis.get("color_season") or analysis.get("body_type"):
        context_bits.append(
            "Client profile: "
            f"color season={analysis.get('color_season') or 'n/a'}, "
            f"undertone={analysis.get('undertone') or 'n/a'}, "
            f"palette={palette or 'n/a'}, "
            f"body type={analysis.get('body_type') or 'n/a'}."
        )
        if analysis.get("body_notes"):
            context_bits.append(str(analysis["body_notes"]))
        if analysis.get("style_direction"):
            context_bits.append(str(analysis["style_direction"]))
    color_q = any(
        p in (user_message or "").lower()
        for p in ("what color", "which color", "suit me", "my palette", "undertone")
    )
    if rec_line and rec_line != "No catalog matches yet." and not color_q:
        context_bits.append(
            "Catalog pieces on the table (prefer palette matches; reject pieces that clash): "
            f"{rec_line}."
        )
    if images:
        context_bits.append(
            f"The shopper attached {len(list(images))} look/garment photo(s). "
            "Read the clothing, color, and silhouette before recommending."
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
            return text, True
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
            return text, True
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

