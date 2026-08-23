"""Export Stylist AI analysis to a downloadable PDF."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image

from .stylist import compact_body_copy, resolve_presentation, swatch_hex


def _register_fonts(pdf: Any) -> str:
    candidates = [
        (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf")),
        (Path(r"C:\Windows\Fonts\segoeui.ttf"), Path(r"C:\Windows\Fonts\segoeuib.ttf")),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.exists():
            pdf.add_font("Vesture", "", str(regular))
            pdf.add_font("Vesture", "B", str(bold if bold.exists() else regular))
            return "Vesture"
    return "Helvetica"


def _hex_rgb(hex_v: str) -> tuple[int, int, int]:
    h = str(hex_v).lstrip("#")
    if len(h) != 6:
        return (139, 92, 246)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _line(pdf: Any, font: str, text: str, *, bold: bool = False, size: int = 11, color=(40, 40, 40), h: float = 6) -> None:
    pdf.set_x(pdf.l_margin)
    pdf.set_font(font, "B" if bold else "", size)
    pdf.set_text_color(*color)
    pdf.multi_cell(pdf.epw, h, text, align="L", new_x="LMARGIN", new_y="NEXT")


def _heading(pdf: Any, font: str, title: str) -> None:
    pdf.ln(4)
    _line(pdf, font, title, bold=True, size=13, color=(25, 25, 25), h=7)


def _jpeg_buf(image: Image.Image, max_side: int) -> io.BytesIO:
    rgb = image.convert("RGB")
    rgb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf


def _swatches(pdf: Any, font: str, names: Sequence[str]) -> None:
    names = [str(n).strip() for n in names if str(n).strip()][:6]
    if not names:
        return
    box, gap, label_w = 8.0, 3.0, 28.0
    row_h = 12.0
    col_w = box + gap + label_w
    cols = max(1, int(pdf.epw // col_w))
    pdf.set_x(pdf.l_margin)
    x0, y = pdf.l_margin, pdf.get_y()
    for i, name in enumerate(names):
        col = i % cols
        row = i // cols
        x = x0 + col * col_w
        yy = y + row * row_h
        if yy + row_h > pdf.h - pdf.b_margin:
            pdf.add_page()
            y = pdf.get_y()
            yy = y
        r, g, b = _hex_rgb(swatch_hex(name))
        pdf.set_fill_color(r, g, b)
        pdf.set_draw_color(200, 200, 200)
        pdf.rect(x, yy, box, box, style="FD")
        pdf.set_xy(x + box + 2, yy + 1.5)
        pdf.set_font(font, "", 8)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(label_w, 5, name[:18], new_x="RIGHT", new_y="TOP")
    rows = (len(names) + cols - 1) // cols
    pdf.set_xy(pdf.l_margin, y + rows * row_h + 2)


def _flow_image(pdf: Any, image: Image.Image, *, width_mm: float, max_side: int = 900) -> None:
    buf = _jpeg_buf(image, max_side)
    peek = Image.open(buf)
    width_mm = min(width_mm, pdf.epw)
    height_mm = width_mm * (peek.height / max(peek.width, 1))
    buf.seek(0)
    if pdf.get_y() + height_mm > pdf.h - pdf.b_margin:
        pdf.add_page()
    pdf.set_x(pdf.l_margin)
    pdf.image(buf, w=width_mm)
    pdf.set_x(pdf.l_margin)
    pdf.ln(3)


def build_analysis_pdf(
    avatar: Image.Image,
    analysis: dict,
    *,
    recs: Optional[Sequence[dict]] = None,
    presentation: str = "",
    color_board: Optional[Image.Image] = None,
    body_board: Optional[Image.Image] = None,
) -> bytes:
    """Return PDF bytes for the current Stylist analysis."""
    from fpdf import FPDF

    analysis = analysis or {}
    recs = list(recs or [])
    dept = resolve_presentation(presentation, analysis)
    dept_label = (
        "Menswear" if dept == "man" else "Womenswear" if dept == "woman" else "Unisex / auto"
    )
    season = str(analysis.get("color_season") or "Unspecified")
    undertone = str(analysis.get("undertone") or "neutral")
    body_type = str(analysis.get("body_type") or "unspecified")
    palette = [str(c) for c in (analysis.get("palette") or []) if str(c).strip()]
    avoid = [str(c) for c in (analysis.get("avoid_colors") or []) if str(c).strip()]

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)
    font = _register_fonts(pdf)
    pdf.add_page()

    _line(pdf, font, "VESTURE Stylist Analysis", bold=True, size=18, color=(20, 20, 20), h=8)
    pdf.ln(2)
    _line(pdf, font, f"{season}  |  {undertone} undertone", bold=True, size=14, color=(30, 30, 30), h=7)
    _line(pdf, font, f"Shop for: {dept_label}", size=11, color=(70, 70, 70))
    pdf.ln(2)

    _flow_image(pdf, avatar, width_mm=42, max_side=640)

    _heading(pdf, font, "01  Body tone")
    notes = str(analysis.get("color_notes") or "").strip()
    if notes:
        _line(pdf, font, notes, size=10, color=(55, 55, 55), h=5)
    _line(pdf, font, f"Body type: {body_type}", size=10, color=(55, 55, 55), h=5)

    _heading(pdf, font, "Recommended color set")
    if palette:
        _swatches(pdf, font, palette)
    else:
        _line(pdf, font, "No palette returned. Try Analyze again with clearer lighting.", size=10)

    if avoid:
        _heading(pdf, font, "Ease off")
        _swatches(pdf, font, avoid)

    _heading(pdf, font, "02  Silhouette")
    body_copy = compact_body_copy(analysis, max_sentences=3)
    if body_copy:
        _line(pdf, font, body_copy, size=10, color=(55, 55, 55), h=5)
    occasions = [str(o) for o in (analysis.get("occasions") or []) if str(o).strip()]
    if occasions:
        _line(pdf, font, "Occasions: " + ", ".join(occasions), size=9, color=(90, 90, 90), h=5)

    pdf.ln(6)
    _line(
        pdf,
        font,
        "Generated by VESTURE Stylist AI. Colors and silhouette are styling guidance, not a medical or identity assessment.",
        size=8,
        color=(130, 130, 130),
        h=4,
    )

    return bytes(pdf.output())
