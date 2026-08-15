"""SegFormer clothing segmentation (mattmdjaga/segformer_b2_clothes)."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from .hf_auth import ensure_hf_login, hf_token
from .preprocess import normalize_garment_region

MODEL_ID = "mattmdjaga/segformer_b2_clothes"

# ATR / model label indices
LABEL_MAP = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Sunglasses",
    4: "Upper-clothes",
    5: "Skirt",
    6: "Pants",
    7: "Dress",
    8: "Belt",
    9: "Left-shoe",
    10: "Right-shoe",
    11: "Face",
    12: "Left-leg",
    13: "Right-leg",
    14: "Left-arm",
    15: "Right-arm",
    16: "Bag",
    17: "Scarf",
}

CATEGORY_TO_LABELS: Dict[str, List[int]] = {
    "upper": [4, 7],
    "lower": [5, 6],
    "dress": [7],
    "upper body": [4, 7],
    "lower body": [5, 6],
}


@lru_cache(maxsize=1)
def load_segformer(device: str | None = None):
    """Load SegFormer processor + model once."""
    from src.hf_auth import ensure_hf_login, hf_token

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_hf_login()
    token = hf_token()
    kw = {"token": token} if token else {}
    processor = SegformerImageProcessor.from_pretrained(MODEL_ID, **kw)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID, **kw)
    model.to(device)
    model.eval()
    return processor, model, device


def _dilate_feather(mask: np.ndarray, dilate_iter: int = 3, sigma: float = 1.5) -> np.ndarray:
    """Morphological dilate + Gaussian feather for cleaner inpainting edges."""
    binary = (mask > 0.5).astype(np.uint8)
    if dilate_iter > 0:
        binary = ndimage.binary_dilation(binary, iterations=dilate_iter).astype(np.float32)
    else:
        binary = binary.astype(np.float32)
    if sigma > 0:
        binary = ndimage.gaussian_filter(binary, sigma=sigma)
    return np.clip(binary, 0.0, 1.0)


def _predict_labels(image: Image.Image) -> Tuple[np.ndarray, torch.Tensor]:
    processor, model, device = load_segformer()
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    probs = torch.softmax(upsampled, dim=1)[0]
    pred = probs.argmax(dim=0).cpu().numpy().astype(np.int32)
    return pred, probs


def _waist_row(pred: np.ndarray) -> int:
    """Row just below the head/torso so a lower mask cannot cover a shirt."""
    h = pred.shape[0]
    face_y = np.where(pred == 11)[0]
    if face_y.size > 12:
        return min(h - 1, int(np.percentile(face_y, 92)) + max(8, int(h * 0.10)))
    hip_y = np.where(np.isin(pred, [5, 6, 12, 13]))[0]
    if hip_y.size > 12:
        return int(np.percentile(hip_y, 8))
    return int(h * 0.45)


def _select_region(pred: np.ndarray, category: str) -> np.ndarray:
    cat = normalize_garment_region(category)
    if cat == "lower":
        selected = np.isin(pred, [5, 6, 8, 12, 13])
        waist = _waist_row(pred)
        selected[:waist, :] = False
        selected[np.isin(pred, [1, 2, 3, 11, 14, 15, 16, 17])] = False
        if selected.mean() < 0.012:
            # Black pants are often tagged as upper-clothes. Keep person pixels below the waist.
            selected = pred != 0
            selected[:waist, :] = False
            selected[np.isin(pred, [1, 2, 3, 9, 10, 11, 14, 15, 16, 17])] = False
        return selected
    if cat == "dress":
        selected = np.isin(pred, [7])
        if not selected.any():
            selected = np.isin(pred, [4, 5, 6, 7])
        return selected
    selected = np.isin(pred, [4, 7])
    if not selected.any():
        selected = np.isin(pred, [4])
    if not selected.any():
        selected = np.isin(pred, [4, 5, 6, 7])
    return selected


def infer_garment_category(image: Image.Image) -> str | None:
    """Guess upper / lower / dress from a garment photo (pants vs shirt vs dress)."""
    pred, _ = _predict_labels(image)
    upper = float(np.isin(pred, [4]).mean())
    lower = float(np.isin(pred, [5, 6]).mean())
    dress = float(np.isin(pred, [7]).mean())
    if lower >= 0.05 and lower >= upper * 0.4:
        return "lower"
    if dress >= 0.10 and dress >= max(upper, lower):
        return "dress"
    if upper >= 0.08 and upper > lower and upper > dress:
        return "upper"
    return None


def segment_clothing(
    image: Image.Image,
    category: str = "upper",
    dilate_iter: int = 3,
    feather_sigma: float = 1.5,
) -> Tuple[Image.Image, float, np.ndarray]:
    """
    Segment clothing region for the given category.

    Returns:
        mask_img: L-mode PIL mask (0-255)
        seg_conf: mean softmax probability over selected clothing pixels
        label_map: HxW int array of predicted labels
    """
    pred, probs = _predict_labels(image)
    selected = _select_region(pred, category)

    if selected.any():
        pixel_conf = probs.max(dim=0).values.cpu().numpy()
        seg_conf = float(pixel_conf[selected].mean())
    else:
        seg_conf = 0.0

    soft = selected.astype(np.float32)
    soft = _dilate_feather(soft, dilate_iter=dilate_iter, sigma=feather_sigma)
    mask_img = Image.fromarray((soft * 255).astype(np.uint8), mode="L")
    return mask_img, seg_conf, pred


def colorize_labels(label_map: np.ndarray) -> Image.Image:
    """Debug visualization of SegFormer labels."""
    palette = np.array(
        [
            [0, 0, 0],
            [128, 0, 0],
            [255, 0, 0],
            [0, 85, 0],
            [85, 85, 0],
            [85, 51, 0],
            [255, 85, 0],
            [255, 170, 0],
            [0, 255, 255],
            [0, 0, 255],
            [0, 119, 221],
            [0, 0, 85],
            [0, 85, 85],
            [85, 85, 127],
            [0, 128, 255],
            [0, 255, 255],
            [85, 51, 127],
            [255, 0, 255],
        ],
        dtype=np.uint8,
    )
    h, w = label_map.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(min(len(palette), int(label_map.max()) + 1)):
        out[label_map == i] = palette[i]
    return Image.fromarray(out, mode="RGB")
