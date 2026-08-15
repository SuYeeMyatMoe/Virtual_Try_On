"""Composite confidence scores and gating (>= 0.85)."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

DEFAULT_GATE = float(os.getenv("CONFIDENCE_GATE", "0.85"))


def mask_quality(mask: Image.Image) -> float:
    """
    Heuristic mask quality in [0, 1]:
    coverage in a sensible range + connectedness of the largest component.
    """
    arr = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    binary = arr > 0.5
    coverage = float(binary.mean())
    # Prefer roughly 5%–55% of image covered by clothing region
    if coverage <= 0:
        cov_score = 0.0
    elif 0.05 <= coverage <= 0.55:
        cov_score = 1.0
    elif coverage < 0.05:
        cov_score = coverage / 0.05
    else:
        cov_score = max(0.0, 1.0 - (coverage - 0.55) / 0.45)

    labeled, n = ndimage.label(binary)
    if n == 0:
        conn_score = 0.0
    else:
        sizes = ndimage.sum(binary, labeled, index=range(1, n + 1))
        largest = float(np.max(sizes))
        conn_score = largest / max(float(binary.sum()), 1.0)

    return float(np.clip(0.5 * cov_score + 0.5 * conn_score, 0.0, 1.0))


def tryon_confidence(
    seg_conf: float,
    clip_sim: float,
    mask: Image.Image,
) -> float:
    """
    tryon_conf = 0.4 * seg_conf + 0.4 * clip_sim + 0.2 * mask_quality
    """
    mq = mask_quality(mask)
    score = 0.4 * float(seg_conf) + 0.4 * float(clip_sim) + 0.2 * mq
    return float(np.clip(score, 0.0, 1.0))


def passes_gate(score: float, gate: float = DEFAULT_GATE) -> bool:
    return float(score) >= float(gate)


def evaluate_segmentation_gate(
    seg_conf: float,
    gate: float = DEFAULT_GATE,
) -> Tuple[bool, str]:
    if passes_gate(seg_conf, gate):
        return True, f"Segmentation confidence {seg_conf:.2%} >= {gate:.0%} gate."
    return (
        False,
        f"Segmentation confidence {seg_conf:.2%} is below the {gate:.0%} gate. "
        "Use a clearer frontal full-body photo with plain background.",
    )


def summarize_scores(
    seg_conf: float,
    clip_sim: float,
    mask: Image.Image,
    gate: float = DEFAULT_GATE,
) -> Dict[str, float | bool | str]:
    mq = mask_quality(mask)
    tryon_conf = tryon_confidence(seg_conf, clip_sim, mask)
    ok_seg, seg_msg = evaluate_segmentation_gate(seg_conf, gate)
    return {
        "seg_conf": float(seg_conf),
        "clip_sim": float(clip_sim),
        "mask_quality": float(mq),
        "tryon_conf": float(tryon_conf),
        "gate": float(gate),
        "passes_seg_gate": bool(ok_seg),
        "passes_tryon_gate": bool(passes_gate(tryon_conf, gate)),
        "seg_message": seg_msg,
    }
