"""Garment-conditioned virtual try-on with Space + SD2 fallbacks.

Primary: IDM-VTON Hugging Face Space (person image + garment image).
Fallback: CatVTON Space, then Stable Diffusion 2 Inpainting, then cached demo.
"""

from __future__ import annotations

import base64
import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import requests
from dotenv import load_dotenv
from PIL import Image

from .hf_auth import ensure_hf_login, hf_token
from .preprocess import build_garment_prompt, negative_prompt

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

DEFAULT_MODEL = os.getenv(
    "HF_INPAINT_MODEL",
    "stabilityai/stable-diffusion-2-inpainting",
)
IDM_SPACE = os.getenv("VTON_SPACE", "yisol/IDM-VTON")
CATVTON_SPACE = os.getenv("CATVTON_SPACE", "zhengchong/CatVTON")
API_URL_TMPL = "https://api-inference.huggingface.co/models/{model}"
DEMO_DIR = Path(__file__).resolve().parents[1] / "assets" / "demo"
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"


def _token() -> str:
    ensure_hf_login()
    token = hf_token()
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add your Hugging Face access token to .env "
            "(vision Inference API only — not an LLM/OpenAI key). See .env.example."
        )
    return token


def _optional_token() -> Optional[str]:
    ensure_hf_login()
    return hf_token()


def _pil_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _as_pil(obj: Any) -> Optional[Image.Image]:
    """Best-effort conversion of Space / API return values to RGB PIL."""
    if obj is None:
        return None
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, (tuple, list)):
        for item in obj:
            img = _as_pil(item)
            if img is not None:
                return img
        return None
    if isinstance(obj, dict):
        for key in ("image", "path", "name", "value", "url"):
            if key in obj:
                img = _as_pil(obj[key])
                if img is not None:
                    return img
        return None
    if isinstance(obj, (str, Path)):
        path = Path(str(obj))
        if path.exists():
            return Image.open(path).convert("RGB")
        return None
    return None


def find_demo_fallback() -> Optional[Image.Image]:
    """Load a cached demo result if present (presentation fallback)."""
    if not DEMO_DIR.exists():
        return None
    for pattern in ("tryon_*.png", "tryon_*.jpg", "*.png", "*.jpg", "*.jpeg", "*.webp"):
        files = sorted(DEMO_DIR.glob(pattern))
        files = [
            f
            for f in files
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            and not f.name.lower().startswith("readme")
        ]
        if files:
            return Image.open(files[0]).convert("RGB")
    return None


def list_demo_pairs() -> list[dict]:
    """VITON-HD-style person/garment pairs under data/samples/."""
    pairs: list[dict] = []
    if not SAMPLES_DIR.exists():
        return pairs
    for person_path in sorted(SAMPLES_DIR.glob("person_*.*")):
        if person_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        stem = person_path.stem.replace("person_", "", 1)
        garment = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            candidate = SAMPLES_DIR / f"cloth_{stem}{ext}"
            if candidate.exists():
                garment = candidate
                break
        if garment is None:
            continue
        meta = SAMPLES_DIR / f"meta_{stem}.txt"
        title = f"VITON-HD pair {stem}"
        category = "upper"
        if meta.exists():
            lines = meta.read_text(encoding="utf-8").strip().splitlines()
            for line in lines:
                if line.lower().startswith("title:"):
                    title = line.split(":", 1)[1].strip() or title
                if line.lower().startswith("category:"):
                    category = line.split(":", 1)[1].strip().lower() or category
        pairs.append(
            {
                "id": stem,
                "title": title,
                "category": category,
                "person_path": person_path,
                "garment_path": garment,
            }
        )
    return pairs


def _save_temp_png(image: Image.Image, folder: Path, name: str) -> str:
    path = folder / name
    image.convert("RGB").save(path, format="PNG")
    return str(path)


def _cloth_type(category: str) -> str:
    cat = (category or "upper").strip().lower()
    if cat in {"dress", "overall", "full"}:
        return "overall"
    if cat in {"lower", "lower body"}:
        return "lower"
    return "upper"


def _gradio_file(path: str):
    try:
        from gradio_client import handle_file

        return handle_file(path)
    except Exception:
        from gradio_client import file as gradio_file

        return gradio_file(path)


def _space_client(space: str, timeout: int = 300):
    from gradio_client import Client

    token = _optional_token()
    kwargs: dict[str, Any] = {}
    if token:
        kwargs["token"] = token
    try:
        return Client(space, **kwargs)
    except TypeError:
        if token:
            kwargs.pop("token", None)
            kwargs["hf_token"] = token
        return Client(space, **kwargs)


def run_idm_vton(
    person: Image.Image,
    garment: Image.Image,
    garment_des: str,
    denoise_steps: int = 30,
    timeout: int = 300,
) -> Image.Image:
    """Call yisol/IDM-VTON Space with person + garment images."""
    client = _space_client(IDM_SPACE, timeout=timeout)
    with tempfile.TemporaryDirectory(prefix="vesture_idm_") as tmp:
        tmp_path = Path(tmp)
        person_path = _save_temp_png(person, tmp_path, "person.png")
        garment_path = _save_temp_png(garment, tmp_path, "garment.png")
        person_file = _gradio_file(person_path)
        out = client.predict(
            {
                "background": person_file,
                "layers": [],
                "composite": person_file,
            },
            garm_img=_gradio_file(garment_path),
            garment_des=garment_des or "fashion garment",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=int(denoise_steps),
            seed=42,
            api_name="/tryon",
        )
    img = _as_pil(out)
    if img is None:
        raise RuntimeError(f"IDM-VTON returned no image: {type(out)}")
    return img


def run_catvton(
    person: Image.Image,
    garment: Image.Image,
    category: str = "upper",
    timeout: int = 300,
) -> Image.Image:
    """Call zhengchong/CatVTON Space with person + garment images."""
    client = _space_client(CATVTON_SPACE, timeout=timeout)
    cloth_type = _cloth_type(category)
    with tempfile.TemporaryDirectory(prefix="vesture_cat_") as tmp:
        tmp_path = Path(tmp)
        person_path = _save_temp_png(person, tmp_path, "person.png")
        garment_path = _save_temp_png(garment, tmp_path, "garment.png")
        person_file = _gradio_file(person_path)
        payload = {
            "background": person_file,
            "layers": [],
            "composite": person_file,
        }
        last_err: Exception | None = None
        for api_name in ("/submit_function", "/predict"):
            try:
                out = client.predict(
                    payload,
                    _gradio_file(garment_path),
                    cloth_type,
                    30,
                    2.5,
                    42,
                    "result only",
                    api_name=api_name,
                )
                img = _as_pil(out)
                if img is not None:
                    return img
                last_err = RuntimeError(f"CatVTON {api_name} returned no image")
            except Exception as exc:
                last_err = exc
                continue
    raise RuntimeError(f"CatVTON failed: {last_err}")


def run_inpainting(
    person: Image.Image,
    mask: Image.Image,
    prompt: str,
    model: str = DEFAULT_MODEL,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    timeout: int = 180,
    max_retries: int = 3,
) -> Image.Image:
    """
    Call HF Inference API for SD2 inpainting.

    mask convention: white = inpaint region, black = keep.
    """
    if mask.mode != "L":
        mask = mask.convert("L")
    if person.mode != "RGB":
        person = person.convert("RGB")
    if person.size != mask.size:
        mask = mask.resize(person.size, Image.Resampling.NEAREST)

    headers = {"Authorization": f"Bearer {_token()}"}
    url = API_URL_TMPL.format(model=model)
    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt(),
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
        },
        "image": _pil_to_b64(person),
        "mask_image": _pil_to_b64(mask),
    }

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            try:
                from huggingface_hub import InferenceClient

                client = InferenceClient(model=model, token=_token(), timeout=timeout)
                if hasattr(client, "image_to_image"):
                    out = client.image_to_image(
                        image=person,
                        prompt=prompt,
                        negative_prompt=negative_prompt(),
                    )
                    if isinstance(out, Image.Image):
                        return out.convert("RGB")
            except Exception as hub_exc:
                last_err = hub_exc

            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 503:
                wait = min(20, 5 * (attempt + 1))
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"HF API {resp.status_code}: {resp.text[:400]}")

            ctype = resp.headers.get("content-type", "")
            if "image" in ctype or resp.content[:8] == b"\x89PNG\r\n\x1a\n":
                return _bytes_to_pil(resp.content)
            try:
                data = resp.json()
            except Exception as json_exc:
                raise RuntimeError(f"Unexpected response: {resp.text[:300]}") from json_exc
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(str(data["error"]))
            raise RuntimeError(f"Unexpected JSON response: {str(data)[:300]}")
        except Exception as exc:
            last_err = exc
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Inpainting failed after retries: {last_err}")


def try_on(
    person: Image.Image,
    mask: Image.Image,
    category: str = "upper",
    color: Optional[str] = None,
    style: Optional[str] = None,
    extra_prompt: Optional[str] = None,
    use_demo_fallback: bool = True,
) -> Tuple[Optional[Image.Image], Optional[str], str]:
    """SD2-only try-on (kept for compatibility). Prefer try_on_vton()."""
    prompt = build_garment_prompt(category, color=color, style=style, extra=extra_prompt)
    try:
        result = run_inpainting(person, mask, prompt)
        return result, None, prompt
    except Exception as exc:
        if use_demo_fallback:
            cached = find_demo_fallback()
            if cached is not None:
                cached = cached.resize(person.size, Image.Resampling.LANCZOS)
                return (
                    cached,
                    f"Inference API unavailable ({exc}). Showing demo fallback image.",
                    prompt,
                )
        return None, str(exc), prompt


def try_on_vton(
    person: Image.Image,
    mask: Image.Image,
    category: str = "upper",
    color: Optional[str] = None,
    style: Optional[str] = None,
    extra_prompt: Optional[str] = None,
    garment: Optional[Image.Image] = None,
    use_demo_fallback: bool = True,
) -> Tuple[Optional[Image.Image], Optional[str], str, str]:
    """
    Garment-conditioned try-on cascade.

    Returns: (result_image or None, warning_or_error or None, prompt_used, engine)
    """
    prompt = extra_prompt or build_garment_prompt(
        category, color=color, style=style, extra=extra_prompt
    )
    errors: list[str] = []

    if garment is not None:
        try:
            result = run_idm_vton(person, garment, prompt)
            return result, None, prompt, "IDM-VTON"
        except Exception as exc:
            errors.append(f"IDM-VTON: {exc}")

        try:
            result = run_catvton(person, garment, category=category)
            warn = "IDM-VTON Space unavailable; used CatVTON."
            if errors:
                warn = f"{warn} ({errors[-1]})"
            return result, warn, prompt, "CatVTON"
        except Exception as exc:
            errors.append(f"CatVTON: {exc}")

    try:
        result = run_inpainting(person, mask, prompt)
        warn = None
        if errors:
            warn = "Garment-conditioned Spaces unavailable; used SD2 inpainting. " + " | ".join(
                errors[-2:]
            )
        elif garment is None:
            warn = "No garment image — used text-guided SD2 inpainting."
        return result, warn, prompt, "SD2 Inpainting"
    except Exception as exc:
        errors.append(f"SD2: {exc}")

    if use_demo_fallback:
        cached = find_demo_fallback()
        if cached is not None:
            cached = cached.resize(person.size, Image.Resampling.LANCZOS)
            return (
                cached,
                "Try-on APIs unavailable. Showing cached demo fallback. " + " | ".join(errors[-3:]),
                prompt,
                "demo fallback",
            )
    return None, " | ".join(errors) or "Try-on failed.", prompt, "none"
