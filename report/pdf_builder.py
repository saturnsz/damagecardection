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
from typing import Any, Dict, List, Optional, Tuple

from fpdf import FPDF
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_PAGE_W = 210  # A4 width  (mm)
_PAGE_H = 297  # A4 height (mm)
_MARGIN = 15
_CONTENT_W = _PAGE_W - 2 * _MARGIN  # usable width
_MAX_IMG_H = 95.0  # cap image height so layout remains compact on A4

_FONT = "Helvetica"
_TITLE_SIZE = 18
_SUBTITLE_SIZE = 12
_SECTION_SIZE = 13
_BODY_SIZE = 10
_SMALL_SIZE = 8

_LINE_H = 6  # default line height for body text
_TABLE_LINE_H = 7

_UNICODE_REPLACEMENTS = {
    "\u2013": "-",   # en-dash
    "\u2014": "-",   # em-dash
    "\u2264": "<=",  # <=
    "\u2265": ">=",  # >=
    "\u2192": "->",  # arrow
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2022": "*",   # bullet
}


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


def _sanitize_latin1(text: str) -> str:
    """Replace unsupported Unicode chars so Helvetica never fails encoding."""
    for src, dst in _UNICODE_REPLACEMENTS.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _safe_str(value: Any) -> str:
    """Convert *value* to latin-1 safe str, returning ``'-'`` for None / empty."""
    if value is None:
        return "-"
    s = _sanitize_latin1(str(value)).strip()
    return s if s else "-"


def _fit_image_box(path: str, max_w: float, max_h: float = _MAX_IMG_H) -> Tuple[float, float]:
    """Compute (width, height) in mm preserving aspect ratio within max_w x max_h."""
    try:
        with Image.open(path) as im:
            px_w, px_h = im.size
        if px_w <= 0 or px_h <= 0:
            return max_w, max_w * 0.75
        ratio = px_h / px_w
        w = max_w
        h = w * ratio
        if h > max_h:
            h = max_h
            w = h / ratio
        return w, h
    except Exception:
        return max_w, min(max_h, max_w * 0.75)


def _ensure_space(pdf: FPDF, needed_h: float) -> None:
    """Add a page if remaining vertical space is less than *needed_h*."""
    if pdf.get_y() + needed_h > _PAGE_H - _MARGIN:
        pdf.add_page()


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

class _ReportPDF(FPDF):
    """Thin FPDF subclass that carries the report-id for the footer."""

    def __init__(self, report_id: str = "", **kwargs):
        super().__init__(**kwargs)
        self._report_id = report_id

    def footer(self):  # noqa: D401
        self.set_y(-12)
        self.set_font(_FONT, "I", _SMALL_SIZE)
        self.set_text_color(140, 140, 140)
        self.cell(0, 8, f"{self._report_id}  |  Halaman {self.page_no()}/{{nb}}", align="C")


_DEFAULT_DISCLAIMER = (
    "Laporan ini merupakan hasil analisis awal sistem otomatis berbasis luas area "
    "kerusakan, tidak mengukur kedalaman atau kerusakan struktural tersembunyi, dan "
    "tidak menggantikan pemeriksaan fisik oleh teknisi atau pemeriksa berwenang."
)


def _fmt_int(value: Any) -> str:
    """Format integer area metric safely."""
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "-"


def build_report(context: Dict[str, Any]) -> bytes:
    """Build the inspection-report PDF and return its content as *bytes*.

    Parameters
    ----------
    context : dict
        Dictionary with keys described in the module docstring / PRD Induk.

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
    dsi: float = float(context.get("dsi", 0.0))
    damage_area: Optional[int] = context.get("damage_area")
    body_area: Optional[int] = context.get("body_area")
    used_fallback_area: bool = bool(context.get("used_fallback_area", False))
    n_clusters: int = int(context.get("n_clusters", 0))
    max_cluster_ratio: float = float(context.get("max_cluster_ratio", 0.0))
    if max_cluster_ratio > 1.0:
        max_cluster_ratio = max_cluster_ratio / 100.0
    triage: Dict[str, Any] = context.get("triage_result", {}) or {}
    severity_label: str = _safe_str(
        context.get("severity_label") or triage.get("label") or triage.get("level_id")
    )
    severity_reason: str = _safe_str(context.get("severity_reason"))
    disclaimer: str = _safe_str(context.get("disclaimer") or _DEFAULT_DISCLAIMER)

    pdf = _ReportPDF(report_id=report_id, orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=_MARGIN)
    pdf.alias_nb_pages()
    pdf.set_margins(_MARGIN, _MARGIN, _MARGIN)
    pdf.add_page()

    # -----------------------------------------------------------------------
    # 1. HEADER  (PD-2)
    # -----------------------------------------------------------------------
    pdf.set_font(_FONT, "B", _TITLE_SIZE)
    pdf.cell(0, 10, "LAPORAN INSPEKSI AWAL", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.set_font(_FONT, "", _SUBTITLE_SIZE)
    pdf.cell(0, 7, "Sistem Deteksi Kerusakan Bodi Mobil", new_x="LMARGIN", new_y="NEXT", align="C")

    pdf.ln(2)
    pdf.set_font(_FONT, "", _SMALL_SIZE)
    pdf.cell(0, 5, f"Report ID: {report_id}   |   Waktu: {timestamp}", new_x="LMARGIN", new_y="NEXT", align="R")

    y_line = pdf.get_y() + 1
    pdf.set_draw_color(60, 60, 60)
    pdf.set_line_width(0.5)
    pdf.line(_MARGIN, y_line, _PAGE_W - _MARGIN, y_line)
    pdf.ln(5)

    # -----------------------------------------------------------------------
    # 2. Kualitas Gambar  (PD-4)
    # -----------------------------------------------------------------------
    _section_heading(pdf, "1. Kualitas Gambar")

    _kv_row(pdf, "Status:", gate_status.upper())
    _kv_row(pdf, "Pesan:", gate_message)

    if gate_status.lower() == "enhanced":
        pdf.set_font(_FONT, "I", _SMALL_SIZE)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(
            0,
            _LINE_H,
            "* Peningkatan kualitas CLAHE telah diterapkan pada gambar.",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_text_color(0, 0, 0)

    pdf.ln(3)

    # -----------------------------------------------------------------------
    # 3. Citra Inspeksi  (PD-1, PD-5)
    # -----------------------------------------------------------------------
    _section_heading(pdf, "2. Citra Inspeksi")

    has_original = _image_exists(original_path)
    has_annotated = _image_exists(annotated_path)
    has_enhanced = _image_exists(enhanced_path)

    img_w_half = (_CONTENT_W - 4) / 2  # 4 mm gap
    img_w_full = _CONTENT_W

    if has_original and has_annotated:
        w1, h1 = _fit_image_box(original_path, img_w_half)
        w2, h2 = _fit_image_box(annotated_path, img_w_half)
        max_h = max(h1, h2)

        _ensure_space(pdf, 7 + max_h + 4)

        pdf.set_font(_FONT, "B", _SMALL_SIZE)
        pdf.cell(img_w_half, 5, "Gambar Asli", new_x="RIGHT", new_y="TOP", align="C")
        pdf.cell(4, 5, "", new_x="RIGHT", new_y="TOP")
        pdf.cell(img_w_half, 5, "Hasil Deteksi", new_x="LMARGIN", new_y="NEXT", align="C")

        y_img = pdf.get_y() + 1
        x1 = _MARGIN + (img_w_half - w1) / 2
        x2 = _MARGIN + img_w_half + 4 + (img_w_half - w2) / 2

        try:
            pdf.image(original_path, x=x1, y=y_img, w=w1)
        except Exception:
            pass
        try:
            pdf.image(annotated_path, x=x2, y=y_img, w=w2)
        except Exception:
            pass

        pdf.set_y(y_img + max_h + 4)
    elif has_original:
        w1, h1 = _fit_image_box(original_path, img_w_full)
        _ensure_space(pdf, 7 + h1 + 4)
        pdf.set_font(_FONT, "B", _SMALL_SIZE)
        pdf.cell(0, 5, "Gambar Asli", new_x="LMARGIN", new_y="NEXT", align="C")
        y_img = pdf.get_y() + 1
        try:
            pdf.image(original_path, x=_MARGIN + (img_w_full - w1) / 2, y=y_img, w=w1)
        except Exception:
            pass
        pdf.set_y(y_img + h1 + 4)
    elif has_annotated:
        w2, h2 = _fit_image_box(annotated_path, img_w_full)
        _ensure_space(pdf, 7 + h2 + 4)
        pdf.set_font(_FONT, "B", _SMALL_SIZE)
        pdf.cell(0, 5, "Hasil Deteksi", new_x="LMARGIN", new_y="NEXT", align="C")
        y_img = pdf.get_y() + 1
        try:
            pdf.image(annotated_path, x=_MARGIN + (img_w_full - w2) / 2, y=y_img, w=w2)
        except Exception:
            pass
        pdf.set_y(y_img + h2 + 4)

    if has_enhanced:
        we, he = _fit_image_box(enhanced_path, img_w_half, max_h=75.0)
        _ensure_space(pdf, 8 + he + 4)
        pdf.ln(2)
        pdf.set_font(_FONT, "B", _SMALL_SIZE)
        pdf.cell(0, 5, "Gambar Setelah Peningkatan (CLAHE)", new_x="LMARGIN", new_y="NEXT", align="C")
        y_img = pdf.get_y() + 1
        try:
            pdf.image(enhanced_path, x=_MARGIN + (_CONTENT_W - we) / 2, y=y_img, w=we)
        except Exception:
            pass
        pdf.set_y(y_img + he + 4)

    pdf.ln(2)

    # -----------------------------------------------------------------------
    # 4. Hasil Deteksi & Perhitungan DSI  (PD-1, PD-4)
    # -----------------------------------------------------------------------
    _ensure_space(pdf, 35)
    _section_heading(pdf, "3. Hasil Deteksi & Perhitungan DSI")

    if detections:
        col_widths = [22, 18, 28, 42, 23, 23]
        total_cw = sum(col_widths)
        scale = _CONTENT_W / total_cw
        col_widths = [round(w * scale, 1) for w in col_widths]

        headers = ["ID", "Klaster", "YOLO Deteksi", "CNN Prediksi", "CNN Conf.", "YOLO Conf."]

        pdf.set_font(_FONT, "B", _SMALL_SIZE)
        pdf.set_fill_color(50, 50, 50)
        pdf.set_text_color(255, 255, 255)
        for i, hdr in enumerate(headers):
            nx = "LMARGIN" if i == len(headers) - 1 else "RIGHT"
            ny = "NEXT" if i == len(headers) - 1 else "TOP"
            pdf.cell(col_widths[i], _TABLE_LINE_H, hdr, border=1, fill=True, align="C", new_x=nx, new_y=ny)
        pdf.set_text_color(0, 0, 0)

        pdf.set_font(_FONT, "", _SMALL_SIZE)
        fill = False
        for det in detections:
            pdf.set_fill_color(235, 235, 235) if fill else pdf.set_fill_color(255, 255, 255)

            row = [
                _safe_str(det.get("ID")),
                _safe_str(det.get("Klaster")),
                _safe_str(det.get("YOLO Deteksi")),
                _safe_str(det.get("CNN Keparahan") or det.get("CNN Prediksi")),
                _fmt_conf(det.get("CNN Confidence")),
                _fmt_conf(det.get("YOLO Confidence")),
            ]
            for i, val in enumerate(row):
                align = "C" if i in (0, 1, 4, 5) else "L"
                nx = "LMARGIN" if i == len(row) - 1 else "RIGHT"
                ny = "NEXT" if i == len(row) - 1 else "TOP"
                pdf.cell(col_widths[i], _TABLE_LINE_H, val, border=1, fill=True, align=align, new_x=nx, new_y=ny)
            fill = not fill
    else:
        pdf.set_font(_FONT, "I", _BODY_SIZE)
        pdf.cell(0, _LINE_H, "Tidak ada kerusakan terdeteksi.", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)

    _ensure_space(pdf, 28)
    pdf.set_font(_FONT, "B", _SECTION_SIZE)
    pdf.set_fill_color(240, 240, 240)
    dsi_text = (
        f"DSI: {dsi:.1f}%   |   Titik Kerusakan: {n_clusters}   |   "
        f"Klaster Dominan: {max_cluster_ratio * 100:.1f}%"
    )
    pdf.cell(0, 10, dsi_text, new_x="LMARGIN", new_y="NEXT", align="C", fill=True, border=1)
    pdf.ln(2)

    if damage_area is not None and body_area is not None:
        area_source = "Cadangan (luas gambar penuh)" if used_fallback_area else "Bounding box bodi kendaraan"
        _kv_row(pdf, "Luas Gabungan Rusak:", f"{_fmt_int(damage_area)} px (union mask)")
        _kv_row(pdf, "Luas Acuan Bodi:", f"{_fmt_int(body_area)} px ({area_source})")

    if used_fallback_area:
        pdf.set_font(_FONT, "I", _SMALL_SIZE)
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(
            0,
            _LINE_H - 1,
            "* Catatan: Bounding box bodi kendaraan tidak tersedia; luas bodi menggunakan luas gambar sebagai cadangan.",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_text_color(0, 0, 0)

    pdf.ln(3)

    # -----------------------------------------------------------------------
    # 5. Penilaian Keparahan & Rekomendasi Triase  (PD-1, TR-4, TR-5)
    # -----------------------------------------------------------------------
    _ensure_space(pdf, 48)
    _section_heading(pdf, "4. Penilaian Keparahan & Rekomendasi Jalur")

    path = _safe_str(triage.get("path"))
    sop = _safe_str(triage.get("sop"))
    estimate = _safe_str(triage.get("estimate"))

    _kv_row(pdf, "Label Keparahan:", severity_label)
    _kv_row(pdf, "Alasan Penilaian:", severity_reason)
    _kv_row(pdf, "Rekomendasi Jalur:", path)
    _kv_row(pdf, "SOP:", sop)
    _kv_row(pdf, "Estimasi:", estimate)

    pdf.ln(4)

    # -----------------------------------------------------------------------
    # 6. Disclaimer  (PD-3)
    # -----------------------------------------------------------------------
    _ensure_space(pdf, 28)
    _section_heading(pdf, "5. Disclaimer")
    pdf.set_font(_FONT, "I", _SMALL_SIZE)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, _LINE_H - 1, disclaimer, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    # -----------------------------------------------------------------------
    # 7. Kolom Tanda Tangan  (PD-7)
    # -----------------------------------------------------------------------
    _ensure_space(pdf, 55)
    _section_heading(pdf, "6. Tanda Tangan")

    sig_w = (_CONTENT_W - 20) / 2
    y_sig = pdf.get_y() + 2

    for i, role in enumerate(["Pemeriksa", "Penerima"]):
        x_col = _MARGIN + i * (sig_w + 20)

        pdf.set_xy(x_col, y_sig)
        pdf.set_font(_FONT, "B", _BODY_SIZE)
        pdf.cell(sig_w, _LINE_H, role, align="C")

        pdf.set_draw_color(0, 0, 0)
        pdf.line(x_col + 10, y_sig + 25, x_col + sig_w - 10, y_sig + 25)

        pdf.set_xy(x_col, y_sig + 27)
        pdf.set_font(_FONT, "", _SMALL_SIZE)
        pdf.cell(sig_w, _LINE_H, "Nama: ____________________", align="C")

        pdf.set_xy(x_col, y_sig + 33)
        pdf.cell(sig_w, _LINE_H, "Tanggal: __________________", align="C")

    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# Internal drawing helpers
# ---------------------------------------------------------------------------

def _section_heading(pdf: FPDF, title: str) -> None:
    """Draw a coloured section heading bar."""
    pdf.set_x(_MARGIN)
    pdf.set_font(_FONT, "B", _SECTION_SIZE)
    pdf.set_fill_color(30, 80, 160)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 8, f"  {_safe_str(title)}", new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _kv_row(pdf: FPDF, key: str, value: str) -> None:
    """Print a bold *key* followed by a normal *value*."""
    pdf.set_x(_MARGIN)
    pdf.set_font(_FONT, "B", _BODY_SIZE)
    pdf.cell(44, _LINE_H, _safe_str(key), new_x="RIGHT", new_y="TOP")
    pdf.set_font(_FONT, "", _BODY_SIZE)
    pdf.multi_cell(0, _LINE_H, _safe_str(value), new_x="LMARGIN", new_y="NEXT")


def _fmt_conf(value: Any) -> str:
    """Format a confidence value as a percentage string."""
    if value is None:
        return "-"
    try:
        v = float(value)
        if v <= 1.0:
            return f"{v * 100:.1f}%"
        return f"{v:.1f}%"
    except (TypeError, ValueError):
        return _safe_str(value)
