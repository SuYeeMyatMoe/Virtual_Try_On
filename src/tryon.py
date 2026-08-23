"""Garment-conditioned virtual try-on with Space + SD2 fallbacks.

Primary: IDM-VTON Hugging Face Space (person image + garment image).
Fallback: CatVTON Space, then Stable Diffusion 2 Inpainting, then a local
garment overlay, then a cached demo image.
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
from .preprocess import (
    build_garment_prompt,
    negative_prompt,
    normalize_color_name,
    normalize_garment_region,
    recolor_hair,
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)

DEFAULT_MODEL = os.getenv(
    "HF_INPAINT_MODEL",
    "stabilityai/stable-diffusion-2-inpainting",
)
IDM_SPACE = os.getenv("VTON_SPACE", "yisol/IDM-VTON")
CATVTON_SPACE = os.getenv("CATVTON_SPACE", "zhengchong/CatVTON")
INPAINT_SPACE = os.getenv("INPAINT_SPACE", "diffusers/stable-diffusion-xl-inpainting")
# api-inference.huggingface.co was decommissioned; Inference Providers live here.
API_URL_TMPL = os.getenv(
    "HF_INFERENCE_URL",
    "https://router.huggingface.co/hf-inference/models/{model}",
)
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
    if isinstance(obj, (bytes, bytearray)):
        try:
            return _bytes_to_pil(bytes(obj))
        except Exception:
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
    cat = normalize_garment_region(category)
    if cat == "dress":
        return "overall"
    return cat


def _prompt_looks_lower(prompt: str) -> bool:
    blob = str(prompt or "").lower()
    return any(
        word in blob
        for word in ("pant", "jean", "skirt", "trouser", "short", "lower-body", "lower body", " a lower")
    )


def _idm_supports(category: str, extra_prompt: str = "") -> bool:
    """Public IDM-VTON Space is VITON-HD upper-body only — pants become tops there."""
    if normalize_garment_region(category) != "upper":
        return False
    if _prompt_looks_lower(extra_prompt):
        return False
    return True


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
    kwargs: dict[str, Any] = {"httpx_kwargs": {"timeout": float(timeout)}}
    if token:
        kwargs["token"] = token
    try:
        return Client(space, **kwargs)
    except TypeError:
        kwargs.pop("httpx_kwargs", None)
        if token:
            kwargs.pop("token", None)
            kwargs["hf_token"] = token
        return Client(space, **kwargs)


def _editor_payload(person_file: Any, layer_file: Any) -> dict[str, Any]:
    """Gradio ImageEditor dict. CatVTON indexes layers[0], so the layer is required."""
    return {
        "background": person_file,
        "layers": [layer_file],
        "composite": person_file,
    }


def _save_mask_layer(
    folder: Path,
    size: tuple[int, int],
    mask: Optional[Image.Image] = None,
) -> str:
    """RGB layer for CatVTON: white = try-on region. Solid black → automasker."""
    if mask is not None:
        layer = mask.convert("L")
        if layer.size != size:
            layer = layer.resize(size, Image.Resampling.NEAREST)
        if layer.getbbox() is not None:
            rgb = Image.merge("RGB", (layer, layer, layer))
            return _save_temp_png(rgb, folder, "mask_layer.png")
    blank = Image.new("RGB", size, (0, 0, 0))
    return _save_temp_png(blank, folder, "mask_layer.png")


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
    mask: Optional[Image.Image] = None,
    timeout: int = 300,
) -> Image.Image:
    """Call zhengchong/CatVTON Space with person + garment + cloth type.

    Live endpoints are /submit_function, /submit_function_flux, /submit_function_p2p.
    /predict is not published on this Space.
    """
    client = _space_client(CATVTON_SPACE, timeout=timeout)
    cloth_type = _cloth_type(category)
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="vesture_cat_") as tmp:
        tmp_path = Path(tmp)
        person_path = _save_temp_png(person, tmp_path, "person.png")
        garment_path = _save_temp_png(garment, tmp_path, "garment.png")
        layer_path = _save_mask_layer(tmp_path, person.size, mask)
        person_file = _gradio_file(person_path)
        cloth_file = _gradio_file(garment_path)
        payload = _editor_payload(person_file, _gradio_file(layer_path))

        # Only fall through to Flux / mask-free if this Space dropped /submit_function.
        # A GPU/queue failure on the main endpoint will fail the others too.
        for api_name in ("/submit_function", "/submit_function_flux"):
            try:
                out = client.predict(
                    person_image=payload,
                    cloth_image=cloth_file,
                    cloth_type=cloth_type,
                    num_inference_steps=30,
                    guidance_scale=2.5,
                    seed=42,
                    show_type="result only",
                    api_name=api_name,
                )
                img = _as_pil(out)
                if img is not None:
                    return img
                errors.append(f"{api_name} returned no image")
            except Exception as exc:
                errors.append(f"{api_name}: {exc}")
                if "cannot find a function" not in str(exc).lower():
                    break

        if errors and all("cannot find a function" in e.lower() for e in errors):
            try:
                out = client.predict(
                    person_image=payload,
                    cloth_image=cloth_file,
                    num_inference_steps=30,
                    guidance_scale=2.5,
                    seed=42,
                    api_name="/submit_function_p2p",
                )
                img = _as_pil(out)
                if img is not None:
                    return img
                errors.append("/submit_function_p2p returned no image")
            except Exception as exc:
                errors.append(f"/submit_function_p2p: {exc}")
    raise RuntimeError(" | ".join(errors) or "CatVTON returned no image")


def run_inpainting(
    person: Image.Image,
    mask: Image.Image,
    prompt: str,
    model: str = DEFAULT_MODEL,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    timeout: int = 60,
    max_retries: int = 2,
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
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 503:
                last_err = RuntimeError(f"HF API 503: {resp.text[:400]}")
                time.sleep(min(20, 5 * (attempt + 1)))
                continue
            if resp.status_code >= 400:
                last_err = RuntimeError(f"HF API {resp.status_code}: {resp.text[:400]}")
                break

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
            if attempt + 1 < max_retries:
                time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Inpainting failed after retries: {last_err}")


def run_inpaint_space(
    person: Image.Image,
    mask: Image.Image,
    prompt: str,
    timeout: int = 180,
) -> Image.Image:
    """Mask inpaint via a Hugging Face Space (works when Inference Providers are 403)."""
    if mask.mode != "L":
        mask = mask.convert("L")
    if person.mode != "RGB":
        person = person.convert("RGB")
    if person.size != mask.size:
        mask = mask.resize(person.size, Image.Resampling.NEAREST)

    client = _space_client(INPAINT_SPACE, timeout=timeout)
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="vesture_hair_") as tmp:
        tmp_path = Path(tmp)
        img_f = _gradio_file(_save_temp_png(person, tmp_path, "person.png"))
        layer_rgba = Image.merge("RGBA", (mask, mask, mask, mask))
        layer_f = _gradio_file(_save_temp_png(layer_rgba, tmp_path, "layer.png"))
        mask_rgb = Image.merge("RGB", (mask, mask, mask))
        mask_f = _gradio_file(_save_temp_png(mask_rgb, tmp_path, "mask.png"))
        editor = _editor_payload(img_f, layer_f)
        neg = negative_prompt()
        attempts = (
            lambda: client.predict(
                editor, prompt, neg, 7.5, 20, 0.85, "EulerDiscreteScheduler", api_name="/predict"
            ),
            lambda: client.predict(img_f, prompt, neg, 7.5, 20, 0.85, "EulerDiscreteScheduler", api_name="/predict"),
            lambda: client.predict(img_f, mask_f, prompt, api_name="/predict"),
            lambda: client.predict(img_f, mask_f, prompt, api_name="/inpaint"),
        )
        for call in attempts:
            try:
                img = _as_pil(call())
                if img is not None:
                    if img.size != person.size:
                        img = img.resize(person.size, Image.Resampling.LANCZOS)
                    return img
                errors.append("Space returned no image")
            except Exception as exc:
                errors.append(str(exc)[:180])
    raise RuntimeError(" | ".join(errors) or f"{INPAINT_SPACE} returned no image")


def run_local_overlay(
    person: Image.Image,
    garment: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """Paste the garment into the masked region when remote try-on APIs are down."""
    person_rgb = person.convert("RGB")
    mask_l = mask.convert("L")
    if mask_l.size != person_rgb.size:
        mask_l = mask_l.resize(person_rgb.size, Image.Resampling.NEAREST)
    bbox = mask_l.getbbox()
    if bbox is None:
        raise RuntimeError("Empty clothing mask — cannot overlay garment.")
    x0, y0, x1, y1 = bbox
    box_w, box_h = max(1, x1 - x0), max(1, y1 - y0)
    fitted = garment.convert("RGB").resize((box_w, box_h), Image.Resampling.LANCZOS)
    region = person_rgb.copy()
    region.paste(fitted, (x0, y0))
    return Image.composite(region, person_rgb, mask_l)


def apply_hair_color(
    person: Image.Image,
    hair_color: str,
    label_map=None,
) -> Tuple[Image.Image, Optional[str]]:
    """Dye only the SegFormer hair region after clothing try-on."""
    named = normalize_color_name(hair_color)
    if not named:
        return person, None
    import importlib

    import src.segmentation as _seg

    importlib.reload(_seg)
    hair_mask = getattr(_seg, "hair_mask", None)
    if hair_mask is None:
        return person, "Hair color skipped — hair mask helper is not loaded. Restart Streamlit."

    mask, _pred = hair_mask(person, label_map=label_map, dilate_iter=1, feather_sigma=2.4)
    if mask.getbbox() is None:
        return person, "Hair color skipped — no hair region found."
    dyed = recolor_hair(person, mask, named)
    return dyed, None


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
    named = normalize_color_name(color)
    prompt = build_garment_prompt(
        category, color=named, extra=extra_prompt
    )
    category = normalize_garment_region(category)
    cloth_type = _cloth_type(category)
    errors: list[str] = []

    if garment is not None:
        # IDM-VTON's public Space has no cloth-type control and always paints
        # onto the torso. Skip it for pants/skirts/dresses so they stay lower-body.
        if _idm_supports(category, prompt):
            try:
                result = run_idm_vton(person, garment, prompt)
                return result, None, prompt, "IDM-VTON"
            except Exception as exc:
                errors.append(f"IDM-VTON: {exc}")

        try:
            result = run_catvton(person, garment, category=category, mask=mask)
            warn = None
            if cloth_type == "upper" and errors:
                warn = f"IDM-VTON Space unavailable; used CatVTON. ({errors[-1]})"
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

    if garment is not None:
        try:
            result = run_local_overlay(person, garment, mask)
            return (
                result,
                "Try-on Spaces and SD2 were unavailable; used a local garment overlay. "
                + " | ".join(errors[-2:]),
                prompt,
                "local overlay",
            )
        except Exception as exc:
            errors.append(f"overlay: {exc}")

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
