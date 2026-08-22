"""Export Stylist AI analysis to a downloadable PDF."""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image

from .stylist import compact_body_copy, resolve_presentation


def _jpeg_bytes(img: Image.Image, *, max_side: int = 1400) -> bytes:
    out = img.convert("RGB")
    out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _register_fonts(pdf: Any) -> tuple[str, str]:
    """Return (regular, bold) font family names registered on this PDF."""
    candidates = [
        (
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\arialbd.ttf"),
        ),
        (
            Path(r"C:\Windows\Fonts\segoeui.ttf"),
            Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
        (
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.exists():
            pdf.add_font("Vesture", "", str(regular))
            pdf.add_font("Vesture", "B", str(bold if bold.exists() else regular))
            return "Vesture", "Vesture"
    return "Helvetica", "Helvetica"


def _set_font(pdf: Any, family: str, style: str, size: int) -> None:
    if family == "Helvetica":
        pdf.set_font(family, style, size)
    else:
        pdf.set_font(family, style, size)


def _section_title(pdf: Any, title: str, *, font_bold: str) -> None:
    _set_font(pdf, font_bold, "B", 13)
    pdf.set_text_color(30, 30, 30)
    pdf.ln(2)
    pdf.multi_cell(0, 7, title)
    pdf.ln(1)


def _body_text(pdf: Any, text: str, *, font_regular: str, size: int = 10) -> None:
    _set_font(pdf, font_regular, "", size)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 5, text or "—")
    pdf.ln(1)


def _fit_image_width(pdf: Any, img: Image.Image, max_w_mm: float) -> tuple[float, float]:
    w_px, h_px = img.size
    if w_px <= 0 or h_px <= 0:
        return max_w_mm, max_w_mm * 0.56
    aspect = h_px / w_px
    w_mm = min(max_w_mm, pdf.w - pdf.l_margin - pdf.r_margin)
    return w_mm, w_mm * aspect


def build_analysis_pdf(
    avatar: Image.Image,
    analysis: dict,
    *,
    color_board: Optional[Image.Image] = None,
    body_board: Optional[Image.Image] = None,
    recs: Optional[Sequence[dict]] = None,
    presentation: str = "",
) -> bytes:
    """Return PDF bytes for the current Stylist analysis."""
    from fpdf import FPDF

    analysis = analysis or {}
    recs = list(recs or [])
    dept = resolve_presentation(presentation, analysis)
    dept_label = (
        "Menswear"
        if dept == "man"
        else "Womenswear"
        if dept == "woman"
        else "Unisex / auto"
    )

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=14)
    font_regular, font_bold = _register_fonts(pdf)
    pdf.add_page()

    _set_font(pdf, font_bold, "B", 20)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 10, "VESTURE · Stylist Analysis", ln=True)
    _set_font(pdf, font_regular, "", 9)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 5, datetime.now().strftime("%d %B %Y · Personal color & silhouette read"), ln=True)
    pdf.ln(4)

    thumb = avatar.convert("RGB").copy()
    thumb.thumbnail((480, 640), Image.Resampling.LANCZOS)
    av_h = min(58.0, 42.0 * (thumb.height / max(thumb.width, 1)))
    av_w = av_h * (thumb.width / max(thumb.height, 1))
    x0 = pdf.w - pdf.r_margin - av_w
    y0 = pdf.get_y()
    pdf.image(io.BytesIO(_jpeg_bytes(thumb, max_side=640)), x=x0, y=y0, w=av_w, h=av_h)

    pdf.set_xy(pdf.l_margin, y0)
    season = str(analysis.get("color_season") or "Unspecified")
    undertone = str(analysis.get("undertone") or "neutral")
    body_type = str(analysis.get("body_type") or "unspecified")
    _set_font(pdf, font_bold, "B", 11)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(x0 - pdf.l_margin - 4, 6, f"{season} · {undertone} undertone")
    _set_font(pdf, font_regular, "", 10)
    pdf.set_text_color(70, 70, 70)
    pdf.multi_cell(x0 - pdf.l_margin - 4, 5, f"Body type: {body_type} · Shop for: {dept_label}")
    pdf.ln(max(av_h - pdf.get_y() + y0 + 2, 4))

    _section_title(pdf, "01 · Body tone", font_bold=font_bold)
    notes = str(analysis.get("color_notes") or "").strip()
    if notes:
        _body_text(pdf, notes, font_regular=font_regular)

    if color_board is not None:
        w_mm, h_mm = _fit_image_width(pdf, color_board, 180)
        if pdf.get_y() + h_mm > pdf.h - pdf.b_margin:
            pdf.add_page()
        pdf.image(io.BytesIO(_jpeg_bytes(color_board)), x=pdf.l_margin, w=w_mm, h=h_mm)
        pdf.ln(2)
        _set_font(pdf, font_regular, "", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 4, "Skin, hair, and cloth sampled from your photo.", ln=True)
        pdf.ln(2)

    _section_title(pdf, "02 · Silhouette", font_bold=font_bold)
    body_copy = compact_body_copy(analysis, max_sentences=3)
    if body_copy:
        _body_text(pdf, body_copy, font_regular=font_regular)
    occasions = [str(o) for o in (analysis.get("occasions") or []) if str(o).strip()]
    if occasions:
        _body_text(pdf, "Occasions: " + ", ".join(occasions), font_regular=font_regular, size=9)

    if body_board is not None:
        w_mm, h_mm = _fit_image_width(pdf, body_board, 180)
        if pdf.get_y() + h_mm > pdf.h - pdf.b_margin:
            pdf.add_page()
        pdf.image(io.BytesIO(_jpeg_bytes(body_board)), x=pdf.l_margin, w=w_mm, h=h_mm)
        pdf.ln(2)
        _set_font(pdf, font_regular, "", 8)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(0, 4, "Estimated from a single photo — not a body scan.", ln=True)
        pdf.ln(2)

    palette = [str(c) for c in (analysis.get("palette") or []) if str(c).strip()]
    avoid = [str(c) for c in (analysis.get("avoid_colors") or []) if str(c).strip()]

    _section_title(pdf, "Recommended color set", font_bold=font_bold)
    if palette:
        _body_text(pdf, ", ".join(palette), font_regular=font_regular)
    else:
        _body_text(
            pdf,
            "No palette returned — try Analyze again with clearer lighting.",
            font_regular=font_regular,
        )

    if avoid:
        _section_title(pdf, "Ease off", font_bold=font_bold)
        _body_text(pdf, ", ".join(avoid), font_regular=font_regular)

    if recs:
        if pdf.get_y() > pdf.h - 60:
            pdf.add_page()
        _section_title(pdf, "Catalog picks for you", font_bold=font_bold)
        _set_font(pdf, font_regular, "", 9)
        pdf.set_text_color(50, 50, 50)
        for i, item in enumerate(recs[:5], 1):
            title = str(item.get("title") or "Item")
            category = str(item.get("category") or "")
            color = str(item.get("color") or "")
            sim = item.get("similarity")
            meta = " · ".join(p for p in (category, color) if p)
            line = f"{i}. {title}"
            if meta:
                line += f" ({meta})"
            if isinstance(sim, (int, float)):
                line += f" — {sim:.0%} match"
            pdf.multi_cell(0, 5, line)
            pdf.ln(0.5)

    _set_font(pdf, font_regular, "", 8)
    pdf.set_text_color(130, 130, 130)
    pdf.ln(4)
    pdf.multi_cell(
        0,
        4,
        "Generated by VESTURE Stylist AI. Colors and silhouette are styling guidance, not medical or identity assessment.",
    )

    out = pdf.output()
    return out if isinstance(out, (bytes, bytearray)) else out.encode("latin-1")
