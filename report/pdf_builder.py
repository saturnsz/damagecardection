"""
pdf_builder.py
==============
Generates an A4 inspection-report PDF for the car body-damage detection system.

Uses fpdf2 (``from fpdf import FPDF``).  Returns raw ``bytes`` suitable for
``st.download_button``.

Public API
----------
- ``generate_report_id()`` -> str
- ``build_report(context: dict)`` -> bytes
- ``cleanup_temp_files(paths: list[str])`` -> None
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fpdf import FPDF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PAGE_W = 210  # A4 width  (mm)
_PAGE_H = 297  # A4 height (mm)
_MARGIN = 15
_CONTENT_W = _PAGE_W - 2 * _MARGIN  # usable width

_FONT = "Helvetica"
_TITLE_SIZE = 18
_SUBTITLE_SIZE = 12
_SECTION_SIZE = 13
_BODY_SIZE = 10
_SMALL_SIZE = 8

_LINE_H = 6  # default line height for body text
_TABLE_LINE_H = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_report_id() -> str:
    """Return a unique report ID in the format ``INS-YYYYMMDD-HHMMSS``."""
    now = datetime.now()
    return f"INS-{now.strftime('%Y%m%d-%H%M%S')}"


def cleanup_temp_files(paths: List[str]) -> None:
    """Silently remove a list of temporary file paths (if they exist)."""
    for p in paths:
        if p and os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _image_exists(path: Optional[str]) -> bool:
    """Return True when *path* is a non-empty string pointing to a real file."""
    return bool(path) and os.path.isfile(path)


def _safe_str(value: Any) -> str:
    """Convert *value* to str, returning ``'-'`` for None / empty."""
    if value is None:
        return "-"
    s = str(value)
    return s if s else "-"


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

class _ReportPDF(FPDF):
    """Thin FPDF subclass that carries the report-id for the footer."""

    def __init__(self, report_id: str = "", **kwargs):
        super().__init__(**kwargs)
        self._report_id = report_id

    # Optional footer with page number
    def footer(self):  # noqa: D401
        self.set_y(-12)
        self.set_font(_FONT, "I", _SMALL_SIZE)
        self.set_text_color(140, 140, 140)
        self.cell(0, 8, f"{self._report_id}  |  Halaman {self.page_no()}/{{nb}}", align="C")


def build_report(context: Dict[str, Any]) -> bytes:
    """Build the inspection-report PDF and return its content as *bytes*.

    Parameters
    ----------
    context : dict
        Dictionary with keys described in the module docstring / PRD.

    Returns
    -------
    bytes
        The finished PDF file content, ready for download.
    """

    report_id: str = _safe_str(context.get("report_id"))
    timestamp: str = _safe_str(context.get("timestamp"))
    original_path: Optional[str] = context.get("original_image_path")
    annotated_path: Optional[str] = context.get("annotated_image_path")
    enhanced_path: Optional[str] = context.get("enhanced_image_path")
    gate_status: str = _safe_str(context.get("gate_status"))
    gate_message: str = _safe_str(context.get("gate_message"))
    detections: List[Dict[str, Any]] = context.get("detections", [])
    dsi: float = context.get("dsi", 0.0)
    triage: Dict[str, Any] = context.get("triage_result", {})
    disclaimer: str = _safe_str(context.get("disclaimer"))

    pdf = _ReportPDF(report_id=report_id, orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=_MARGIN)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_margins(_MARGIN, _MARGIN, _MARGIN)

    # -----------------------------------------------------------------------
    # 1. HEADER  (PD-2)
    # -----------------------------------------------------------------------
    pdf.set_font(_FONT, "B", _TITLE_SIZE)
    pdf.cell(0, 10, "LAPORAN INSPEKSI AWAL", ln=True, align="C")

    pdf.set_font(_FONT, "", _SUBTITLE_SIZE)
    pdf.cell(0, 7, "Sistem Deteksi Kerusakan Bodi Mobil", ln=True, align="C")

    pdf.ln(2)
    pdf.set_font(_FONT, "", _SMALL_SIZE)
    pdf.cell(0, 5, f"Report ID: {report_id}   |   Waktu: {timestamp}", ln=True, align="R")

    # horizontal rule
    y_line = pdf.get_y() + 1
    pdf.set_draw_color(60, 60, 60)
    pdf.set_line_width(0.5)
    pdf.line(_MARGIN, y_line, _PAGE_W - _MARGIN, y_line)
    pdf.ln(5)

    # -----------------------------------------------------------------------
    # 2. Kualitas Gambar  (PD-4)
    # -----------------------------------------------------------------------
    _section_heading(pdf, "1. Kualitas Gambar")

    pdf.set_font(_FONT, "B", _BODY_SIZE)
    pdf.cell(35, _LINE_H, "Status:")
    pdf.set_font(_FONT, "", _BODY_SIZE)
    pdf.cell(0, _LINE_H, gate_status.upper(), ln=True)

    pdf.set_font(_FONT, "B", _BODY_SIZE)
    pdf.cell(35, _LINE_H, "Pesan:")
    pdf.set_font(_FONT, "", _BODY_SIZE)
    pdf.multi_cell(0, _LINE_H, gate_message)

    if gate_status.lower() == "enhanced":
        pdf.set_font(_FONT, "I", _SMALL_SIZE)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, _LINE_H, "* Peningkatan kualitas CLAHE telah diterapkan pada gambar.", ln=True)
        pdf.set_text_color(0, 0, 0)

    pdf.ln(3)

    # -----------------------------------------------------------------------
    # 3. Citra Inspeksi  (PD-1, PD-5)
    # -----------------------------------------------------------------------
    _section_heading(pdf, "2. Citra Inspeksi")

    has_original = _image_exists(original_path)
    has_annotated = _image_exists(annotated_path)
    has_enhanced = _image_exists(enhanced_path)

    # Decide image width — two side-by-side when both exist
    img_w_half = (_CONTENT_W - 4) / 2  # 4 mm gap
    img_w_full = _CONTENT_W

    if has_original and has_annotated:
        x_start = pdf.get_x()
        y_start = pdf.get_y()

        # Label + image — Original
        pdf.set_font(_FONT, "B", _SMALL_SIZE)
        pdf.cell(img_w_half, 5, "Gambar Asli", ln=False, align="C")
        pdf.cell(4, 5, "")  # gap
        pdf.cell(img_w_half, 5, "Hasil Deteksi", ln=True, align="C")

        y_img = pdf.get_y()
        try:
            pdf.image(original_path, x=_MARGIN, y=y_img, w=img_w_half)
        except Exception:
            pass
        try:
            info = pdf.image(annotated_path, x=_MARGIN + img_w_half + 4, y=y_img, w=img_w_half)
        except Exception:
            info = None

        # Advance Y past whichever image is taller
        # fpdf2 image() returns an object; we estimate height from width ratio
        img_h_est = img_w_half * 0.75  # 4:3 fallback
        pdf.set_y(y_img + img_h_est + 3)
    elif has_original:
        pdf.set_font(_FONT, "B", _SMALL_SIZE)
        pdf.cell(0, 5, "Gambar Asli", ln=True, align="C")
        try:
            pdf.image(original_path, x=_MARGIN, w=img_w_full)
        except Exception:
            pass
        pdf.ln(3)
    elif has_annotated:
        pdf.set_font(_FONT, "B", _SMALL_SIZE)
        pdf.cell(0, 5, "Hasil Deteksi", ln=True, align="C")
        try:
            pdf.image(annotated_path, x=_MARGIN, w=img_w_full)
        except Exception:
            pass
        pdf.ln(3)

    # Enhanced image (full width, below the pair)
    if has_enhanced:
        pdf.ln(2)
        pdf.set_font(_FONT, "B", _SMALL_SIZE)
        pdf.cell(0, 5, "Gambar Setelah Peningkatan (CLAHE)", ln=True, align="C")
        try:
            pdf.image(enhanced_path, x=_MARGIN + img_w_half / 2, w=img_w_half)
        except Exception:
            pass
        pdf.ln(3)

    pdf.ln(3)

    # -----------------------------------------------------------------------
    # 4. Hasil Deteksi  (PD-1)
    # -----------------------------------------------------------------------
    _section_heading(pdf, "3. Hasil Deteksi")

    if detections:
        col_widths = [12, 42, 38, 32, 32]  # must sum ~ _CONTENT_W  (156 ~= 180-24 padded)
        # Adjust to fill content width
        total_cw = sum(col_widths)
        scale = _CONTENT_W / total_cw
        col_widths = [round(w * scale, 1) for w in col_widths]

        headers = ["ID", "YOLO Deteksi", "CNN Keparahan", "CNN Conf.", "YOLO Conf."]

        # Table header
        pdf.set_font(_FONT, "B", _SMALL_SIZE)
        pdf.set_fill_color(50, 50, 50)
        pdf.set_text_color(255, 255, 255)
        for i, hdr in enumerate(headers):
            pdf.cell(col_widths[i], _TABLE_LINE_H, hdr, border=1, fill=True, align="C")
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        # Table rows
        pdf.set_font(_FONT, "", _SMALL_SIZE)
        fill = False
        for det in detections:
            if fill:
                pdf.set_fill_color(235, 235, 235)
            else:
                pdf.set_fill_color(255, 255, 255)

            row = [
                _safe_str(det.get("ID")),
                _safe_str(det.get("YOLO Deteksi")),
                _safe_str(det.get("CNN Keparahan")),
                _fmt_conf(det.get("CNN Confidence")),
                _fmt_conf(det.get("YOLO Confidence")),
            ]
            for i, val in enumerate(row):
                align = "C" if i == 0 else ("L" if i <= 2 else "C")
                pdf.cell(col_widths[i], _TABLE_LINE_H, val, border=1, fill=True, align=align)
            pdf.ln()
            fill = not fill
    else:
        pdf.set_font(_FONT, "I", _BODY_SIZE)
        pdf.cell(0, _LINE_H, "Tidak ada kerusakan terdeteksi.", ln=True)

    pdf.ln(3)

    # DSI
    pdf.set_font(_FONT, "B", _SECTION_SIZE)
    pdf.set_fill_color(240, 240, 240)
    dsi_text = f"Damage Severity Index (DSI): {dsi:.1f}%"
    pdf.cell(0, 10, dsi_text, ln=True, align="C", fill=True, border=1)
    pdf.ln(4)

    # -----------------------------------------------------------------------
    # 5. Rekomendasi Triase  (PD-1, TR-5)
    # -----------------------------------------------------------------------
    _section_heading(pdf, "4. Rekomendasi Jalur")

    level = _safe_str(triage.get("level"))
    level_id = _safe_str(triage.get("level_id"))
    path = _safe_str(triage.get("path"))
    sop = _safe_str(triage.get("sop"))
    estimate = _safe_str(triage.get("estimate"))
    override = triage.get("override", False)
    override_reason = _safe_str(triage.get("override_reason"))
    original_level = _safe_str(triage.get("original_level"))

    _kv_row(pdf, "Level:", f"{level} ({level_id})")
    _kv_row(pdf, "Jalur:", path)
    _kv_row(pdf, "SOP:", sop)
    _kv_row(pdf, "Estimasi:", estimate)

    if override:
        pdf.ln(2)
        pdf.set_font(_FONT, "B", _BODY_SIZE)
        pdf.set_text_color(180, 0, 0)
        pdf.cell(0, _LINE_H, "* Override diterapkan", ln=True)
        pdf.set_text_color(0, 0, 0)
        _kv_row(pdf, "Level Awal:", original_level)
        _kv_row(pdf, "Alasan Override:", override_reason)

    pdf.ln(4)

    # -----------------------------------------------------------------------
    # 6. Disclaimer  (PD-3)
    # -----------------------------------------------------------------------
    _section_heading(pdf, "5. Disclaimer")
    pdf.set_font(_FONT, "I", _SMALL_SIZE)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, _LINE_H - 1, disclaimer)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # -----------------------------------------------------------------------
    # 7. Kolom Tanda Tangan  (PD-7)
    # -----------------------------------------------------------------------
    _section_heading(pdf, "6. Tanda Tangan")

    sig_w = (_CONTENT_W - 20) / 2  # two columns with a gap
    y_sig = pdf.get_y() + 2

    # Ensure enough space for signature block; force new page if near bottom
    if y_sig + 40 > _PAGE_H - _MARGIN:
        pdf.add_page()
        y_sig = pdf.get_y() + 2

    for i, role in enumerate(["Pemeriksa", "Penerima"]):
        x_col = _MARGIN + i * (sig_w + 20)

        pdf.set_xy(x_col, y_sig)
        pdf.set_font(_FONT, "B", _BODY_SIZE)
        pdf.cell(sig_w, _LINE_H, role, align="C", ln=True)

        # space for signature
        pdf.set_xy(x_col, y_sig + 25)
        pdf.set_draw_color(0, 0, 0)
        pdf.line(x_col + 10, y_sig + 25, x_col + sig_w - 10, y_sig + 25)

        pdf.set_xy(x_col, y_sig + 27)
        pdf.set_font(_FONT, "", _SMALL_SIZE)
        pdf.cell(sig_w, _LINE_H, "Nama: ____________________", align="C")

        pdf.set_xy(x_col, y_sig + 33)
        pdf.cell(sig_w, _LINE_H, "Tanggal: __________________", align="C")

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Internal drawing helpers
# ---------------------------------------------------------------------------

def _section_heading(pdf: FPDF, title: str) -> None:
    """Draw a coloured section heading bar."""
    pdf.set_font(_FONT, "B", _SECTION_SIZE)
    pdf.set_fill_color(30, 80, 160)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, f"  {title}", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _kv_row(pdf: FPDF, key: str, value: str) -> None:
    """Print a bold *key* followed by a normal *value*."""
    pdf.set_font(_FONT, "B", _BODY_SIZE)
    pdf.cell(35, _LINE_H, key)
    pdf.set_font(_FONT, "", _BODY_SIZE)
    pdf.multi_cell(0, _LINE_H, value)


def _fmt_conf(value: Any) -> str:
    """Format a confidence value as a percentage string."""
    if value is None:
        return "-"
    try:
        v = float(value)
        # If already 0-1 range, convert to percentage
        if v <= 1.0:
            return f"{v * 100:.1f}%"
        return f"{v:.1f}%"
    except (TypeError, ValueError):
        return _safe_str(value)
