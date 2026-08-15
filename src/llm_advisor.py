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
    "to try first. When the shopper shares a photo, read the garment, color, and fit before advising. "
    "When they send a voice note, treat the transcript as their brief. "
    "If a client profile exists, stay inside that palette and body type. "
    "If nothing has been analyzed yet, still give precise general advice. "
    "Reply in 2–6 short sentences. Do not invent brand names. No hashtags. No emoji.\n"
    "Example — shopper: I have a garden wedding and cool undertones.\n"
    "You: Stay in jewel and icy tones — emerald, powder blue, true navy. "
    "A defined-waist midi or a tailored jacket over a fluid skirt will photograph well "
    "and keep the silhouette formal without feeling costume.\n"
    "Example — shopper: Casual weekend, pear shape.\n"
    "You: Structure the top — a collared shirt or cropped jacket — and keep trousers "
    "in a continuous dark or mid tone so the hip line stays quiet. Add one leather or "
    "metal accessory so the look does not read flat."
)

FEW_SHOT_CAPTION = (
    "Example: a navy cotton crew-neck t-shirt with a relaxed fit and matte finish.\n"
    "Example: a black tailored blazer in structured wool with notch lapels."
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
    bits = [p for p in (color, style, category) if p]
    look = " ".join(bits) if bits else "fashion garment"
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
    prompt = (
        f"{FEW_SHOT_CAPTION}\n"
        "Describe this garment in one sentence for a virtual try-on model. "
        "Include color, fabric, silhouette, and garment type. "
        f"Category hint: {category}."
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
        if isinstance(sim, (int, float)):
            parts.append(f"{title} ({sim:.0%})")
        else:
            parts.append(str(title))
    return "; ".join(parts)


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
    "Return JSON only."
)


def analyze_avatar_llm(image: Image.Image) -> tuple[dict, bool]:
    """Gemini color + body + style JSON. Returns (payload, used_gemini)."""
    prompt = (
        "Analyze this person for personal styling. Return JSON with keys:\n"
        "color_season (string, e.g. Soft Autumn),\n"
        "undertone (warm|cool|neutral),\n"
        "palette (array of 5 wearable color names),\n"
        "avoid_colors (array of 3 color names),\n"
        "color_notes (2 sentences on skin/hair/eye contrast),\n"
        "body_type (hourglass|pear|apple|inverted triangle|rectangle|athletic|unspecified),\n"
        "body_notes (2 sentences on silhouette; say unspecified if the crop is not full body),\n"
        "style_direction (2 sentences of style recommendation),\n"
        "silhouette_tips (1–2 sentences),\n"
        "occasions (array of 2–3 occasions).\n"
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
        if analysis.get("style_direction"):
            context_bits.append(str(analysis["style_direction"]))
    if rec_line and rec_line != "No catalog matches yet.":
        context_bits.append(f"Catalog pieces on the table: {rec_line}.")
    if images:
        context_bits.append(
            f"The shopper attached {len(list(images))} look/garment photo(s). "
            "Read the clothing, color, and silhouette before recommending."
        )
    if audio_bytes:
        context_bits.append("The shopper also sent a voice note; honor that brief.")
    context = "\n".join(context_bits)
    fallback = (
        "I can help with outfits, colors, and occasions. "
        "Upload an avatar and tap Analyze for a personal color and body reading, "
        "or ask me anything about styling."
    )
    if rec_line and rec_line != "No catalog matches yet.":
        fallback = f"{fallback} Closest catalog pieces: {rec_line}."
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
        return text, True
    except Exception:
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

