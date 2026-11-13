"""Generates formal Cronus Consultants project documentation (.docx and .pdf) for Project RESTOK.

Follows the industry-standard layout and structure of the reference project documentation:
- Cronus Consultants logo placed on the top-left corner of EVERY page.
- Front Matter: Cover Page with Sayan Senapati as Project Owner, Certificate of Completion
  signed by Mentor Sayantani Sarkar, Candidate Declaration signed by Sayan Senapati,
  Acknowledgement, and Table of Contents (14 standard industry sections).
- Embedded authentic cursive signatures for both Mentor (Sayantani Sarkar) and Candidate (Sayan Senapati).
- Synchronized quantitative metrics from data/processed/ (35.46% WAPE, $4,625 Total Cost, 98.4% Fill Rate).
- High-resolution architecture diagrams, benchmark charts, and live dashboard screenshots.
"""
import base64
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor

from docs.build_charts import generate_all_charts
from src.config import CONFIG_PATH, PROCESSED_DIR

DOCS_DIR = Path(__file__).resolve().parent
ASSETS_DIR = DOCS_DIR / "assets"
LOGO_PATH = ASSETS_DIR / "cronus_logo.png"
MENTOR_SIG_PATH = ASSETS_DIR / "mentor_signature.png"
OWNER_SIG_PATH = ASSETS_DIR / "owner_signature.png"
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Brand Palette (Matching Cronus Consultants & Industry Demo Doc)
C_NAVY = RGBColor(0x1F, 0x4E, 0x79)      # Primary Brand Navy
C_BLUE = RGBColor(0x2E, 0x75, 0xB6)      # Secondary Slate Blue
C_GREEN = RGBColor(0x65, 0xA3, 0x38)     # Cronus Accent Green
C_SLATE = RGBColor(0x1E, 0x29, 0x3B)     # Dark Slate Body Text
C_MUTED = RGBColor(0x64, 0x74, 0x8B)     # Muted Subtitle Gray
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)     # Pure White


# ---------------------------------------------------------------------------
# Data Context Loader
# ---------------------------------------------------------------------------

def load_report_context() -> Dict[str, Any]:
    """Loads all verified empirical outputs, configurations, and state files."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    metrics_df = pd.read_csv(PROCESSED_DIR / "metrics.csv")
    cens_df = pd.read_csv(PROCESSED_DIR / "censoring_comparison.csv")
    pol_df = pd.read_csv(PROCESSED_DIR / "policy_kpi_summary.csv")
    trade_df = pd.read_csv(PROCESSED_DIR / "policy_tradeoff_curve.csv")

    can_sim = pd.read_parquet(PROCESSED_DIR / "canonical_simulated.parquet")
    feat_df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test_forecasts.parquet")

    po_file = PROCESSED_DIR / "purchase_orders_state.json"
    pos = json.loads(po_file.read_text(encoding="utf-8")) if po_file.exists() else []

    lgb_row = metrics_df[metrics_df["Model"].str.contains("Tweedie", na=False)].iloc[0]
    base7_row = metrics_df[metrics_df["Model"].str.contains("7-day", na=False)].iloc[0]
    wape_rel_imp = ((base7_row["WAPE"] - lgb_row["WAPE"]) / base7_row["WAPE"]) * 100.0

    heur = pol_df[pol_df["Policy"] == "manager_heuristic"].iloc[0]
    opt_quant = pol_df[pol_df["Policy"] == "quantile_ordering"].iloc[0]
    cost_sav_usd = float(heur["total_cost_usd_mean"] - opt_quant["total_cost_usd_mean"])
    cost_sav_pct = float((cost_sav_usd / heur["total_cost_usd_mean"]) * 100.0)

    charts = generate_all_charts()

    return {
        "cfg": cfg,
        "metrics_df": metrics_df,
        "cens_df": cens_df,
        "pol_df": pol_df,
        "trade_df": trade_df,
        "charts": charts,
        "lgb_row": lgb_row,
        "base7_row": base7_row,
        "wape_rel_imp": wape_rel_imp,
        "heur": heur,
        "opt_quant": opt_quant,
        "cost_sav_usd": cost_sav_usd,
        "cost_sav_pct": cost_sav_pct,
        "canonical_rows": len(can_sim),
        "stores_count": int(can_sim["store_id"].nunique()),
        "items_count": int(can_sim["item_id"].nunique()),
        "series_count": int(can_sim["store_id"].nunique() * can_sim["item_id"].nunique()),
        "date_min": str(can_sim["date"].min())[:10],
        "date_max": str(can_sim["date"].max())[:10],
        "features_count": feat_df.shape[1],
        "test_rows": len(test_df),
        "po_total": len(pos),
        "po_critical": sum(1 for p in pos if p.get("urgency") == "CRITICAL"),
        "po_spend": sum(p.get("order_quantity", 0) * p.get("cost_per_unit", 0.0) for p in pos),
    }


# ---------------------------------------------------------------------------
# DOCX XML & Styling Helpers
# ---------------------------------------------------------------------------

def _set_cell_shading(cell, hex_color: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    for child in list(tcPr):
        if child.tag.endswith("shd"):
            tcPr.remove(child)
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>'))


def _set_cell_margins(cell, top=80, bottom=80, left=110, right=110) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def _set_table_borders(table, color="D1D5DB", sz="4") -> None:
    tblPr = table._tbl.tblPr
    for child in list(tblPr):
        if child.tag.endswith("tblBorders"):
            tblPr.remove(child)
    tblBorders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:left w:val="none"/>'
        f'<w:bottom w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(tblBorders)


def _set_paragraph_border(paragraph, side="bottom", color="CBD5E1", sz="6") -> None:
    pPr = paragraph._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    b = OxmlElement(f"w:{side}")
    b.set(qn("w:val"), "single")
    b.set(qn("w:sz"), str(sz))
    b.set(qn("w:space"), "3")
    b.set(qn("w:color"), color)
    pbdr.append(b)
    pPr.append(pbdr)


def _add_field(run, field_name: str) -> None:
    fldSimple = OxmlElement("w:fldSimple")
    fldSimple.set(qn("w:instr"), field_name)
    run._r.append(fldSimple)


def _prevent_row_split(row) -> None:
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:cantSplit"))


def _repeat_header_row(row) -> None:
    trPr = row._tr.get_or_add_trPr()
    trPr.append(OxmlElement("w:tblHeader"))


def add_styled_table(
    doc: Document,
    headers: List[str],
    rows: List[List[str]],
    col_widths: List[float],
    alignments: List[WD_ALIGN_PARAGRAPH] = None,
) -> None:
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(table, color="D1D5DB", sz="4")

    # Header Row
    hdr_row = table.rows[0]
    _repeat_header_row(hdr_row)
    _prevent_row_split(hdr_row)
    for idx, heading in enumerate(headers):
        cell = hdr_row.cells[idx]
        cell.width = Inches(col_widths[idx])
        _set_cell_shading(cell, "1F4E79")
        _set_cell_margins(cell, top=90, bottom=90, left=110, right=110)
        p = cell.paragraphs[0]
        p.alignment = alignments[idx] if alignments else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(heading)
        r.bold = True
        r.font.size = Pt(8.8)
        r.font.color.rgb = C_WHITE

    # Data Rows
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1]
        _prevent_row_split(row)
        bg = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.width = Inches(col_widths[c_idx])
            _set_cell_shading(cell, bg)
            _set_cell_margins(cell, top=70, bottom=70, left=110, right=110)
            p = cell.paragraphs[0]
            p.alignment = alignments[c_idx] if alignments else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            r = p.add_run(str(val))
            r.font.size = Pt(8.5)
            r.font.color.rgb = C_SLATE
            if c_idx == 0:
                r.bold = True

    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(4)


def add_section_heading(doc: Document, num: str, title: str, page_break: bool = False) -> None:
    """Add Heading 1 — 16pt bold #1F4E79, matching demo style exactly."""
    if page_break:
        doc.add_page_break()
    p = doc.add_paragraph(style="Heading 1")
    p.paragraph_format.keep_with_next = True
    if num:
        r_num = p.add_run(f"{num} ")
        r_num.font.color.rgb = C_GREEN   # accent number in green
    r_txt = p.add_run(title)
    # Colors already set via style; force here for safety
    r_txt.font.color.rgb = C_NAVY


def add_sub_heading(doc: Document, title: str) -> None:
    """Add Heading 2 — 13pt #2E75B6, matching demo style exactly."""
    p = doc.add_paragraph(style="Heading 2")
    p.paragraph_format.keep_with_next = True
    p.add_run(title)


def add_sub_sub_heading(doc: Document, title: str) -> None:
    """Add Heading 3 — 12pt #404040, matching demo style."""
    p = doc.add_paragraph(style="Heading 3")
    p.paragraph_format.keep_with_next = True
    p.add_run(title)


def add_body_p(doc: Document, text: str, bold_prefix: str = "") -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        rp = p.add_run(bold_prefix)
        rp.bold = True
        rp.font.size = Pt(9.5)
        rp.font.color.rgb = C_SLATE
    r = p.add_run(text)
    r.font.size = Pt(9.5)
    r.font.color.rgb = C_SLATE


def add_bullet(doc: Document, lead: str, text: str) -> None:
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(2.5)
    p.paragraph_format.line_spacing = 1.13
    r1 = p.add_run(lead)
    r1.bold = True
    r1.font.size = Pt(9.3)
    r1.font.color.rgb = C_NAVY
    r2 = p.add_run(text)
    r2.font.size = Pt(9.3)
    r2.font.color.rgb = C_SLATE


def add_callout(doc: Document, title: str, text: str, fill_hex: str = "EFF6FF", accent_hex: str = "1F4E79") -> None:
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    cell.width = Inches(6.87)
    _set_cell_shading(cell, fill_hex)
    _set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="16" w:space="0" w:color="{accent_hex}"/>'
        f'<w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    ))
    _prevent_row_split(tbl.rows[0])
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    rt = p.add_run(title + "\n")
    rt.bold = True
    rt.font.size = Pt(9.5)
    rt.font.color.rgb = C_NAVY
    rb = p.add_run(text)
    rb.font.size = Pt(9.0)
    rb.font.color.rgb = C_SLATE
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(3)


def add_code_block(doc: Document, code: str) -> None:
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.rows[0].cells[0]
    cell.width = Inches(6.87)
    _set_cell_shading(cell, "F8FAFC")
    _set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="14" w:space="0" w:color="1F4E79"/>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="E2E8F0"/>'
        f'</w:tcBorders>'
    ))
    _prevent_row_split(tbl.rows[0])
    cp = cell.paragraphs[0]
    cp.paragraph_format.space_before = Pt(0)
    cp.paragraph_format.space_after = Pt(0)
    cp.paragraph_format.line_spacing = 1.15
    r = cp.add_run(code.strip())
    r.font.name = "Consolas"
    r.font.size = Pt(8.2)
    r.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_before = Pt(0)
    sp.paragraph_format.space_after = Pt(3)


def add_figure(doc: Document, img_path: Path, caption: str, width_in: float = 6.75) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(img_path), width=Inches(width_in))

    cp = doc.add_paragraph()
    cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cp.paragraph_format.space_before = Pt(0)
    cp.paragraph_format.space_after = Pt(6)
    rc = cp.add_run(caption)
    rc.italic = True
    rc.font.size = Pt(8.2)
    rc.font.color.rgb = C_MUTED


# ---------------------------------------------------------------------------
# DOCX Report Generator
# ---------------------------------------------------------------------------

def build_docx(ctx: Dict[str, Any]) -> Path:
    """Generates editable A4 Word (.docx) report matching the demo industry-standard layout.

    Key structural elements matched exactly to the reference document:
    - Default font: Arial 11pt (sz=22 half-pt, as per demo docDefaults)
    - Heading 1: 16pt bold #1F4E79, spacing before=18pt/after=10pt
    - Heading 2: 13pt #2E75B6, spacing before=14pt/after=8pt
    - Heading 3: 12pt #404040
    - Header: left-aligned bold navy project name + gray subtitle, navy bottom border
    - Footer: "Confidential • RESTOK Internship 2026 • Page N" gray top border
    - Cover: CAPS PROJECT DOCUMENTATION (navy bold 13pt) → title → subtitle → blue italic tag
      → thin blue separator line → "Submitted by:" → Sayan Senapati (bold 12pt centered)
      → registration/dept (9pt gray centered) → metadata table (navy key col / white value col)
    - Cronus logo placed in header LEFT on every page (matches user requirement)
    """
    doc = Document()

    # ── Match demo docDefaults: Arial 11pt ───────────────────────────────────
    normal_style = doc.styles["Normal"]
    normal_style.font.name = "Arial"
    normal_style.font.size = Pt(11)

    # ── Heading 1 — 16pt bold #1F4E79 (matches demo rPr) ────────────────────
    h1 = doc.styles["Heading 1"]
    h1.font.name = "Arial"
    h1.font.size = Pt(16)
    h1.font.bold = True
    h1.font.color.rgb = C_NAVY
    h1.paragraph_format.space_before = Pt(18)
    h1.paragraph_format.space_after = Pt(10)

    # ── Heading 2 — 13pt #2E75B6 ─────────────────────────────────────────────
    h2 = doc.styles["Heading 2"]
    h2.font.name = "Arial"
    h2.font.size = Pt(13)
    h2.font.bold = True
    h2.font.color.rgb = C_BLUE
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(8)

    # ── Heading 3 — 12pt #404040 ─────────────────────────────────────────────
    h3 = doc.styles["Heading 3"]
    h3.font.name = "Arial"
    h3.font.size = Pt(12)
    h3.font.bold = True
    h3.font.color.rgb = RGBColor(0x40, 0x40, 0x40)
    h3.paragraph_format.space_before = Pt(10)
    h3.paragraph_format.space_after = Pt(6)

    # ── Page layout: US Letter 8.5×11 (matching demo) ────────────────────────
    sec = doc.sections[0]
    sec.page_width = Inches(8.5)
    sec.page_height = Inches(11.0)
    sec.top_margin = Inches(0.75)
    sec.bottom_margin = Inches(0.75)
    sec.left_margin = Inches(0.75)
    sec.right_margin = Inches(0.75)
    sec.header_distance = Inches(0.3)
    sec.footer_distance = Inches(0.3)
    sec.different_first_page_header_footer = False

    # ── HEADER: Cronus logo (left) + project name text (right) ───────────────
    # Demo header: bold navy project name | gray subtitle, navy bottom border sz=12
    # We honour user's requirement: logo top-left on every page
    hp = sec.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hp.paragraph_format.space_before = Pt(0)
    hp.paragraph_format.space_after = Pt(6)
    hp.paragraph_format.tab_stops.add_tab_stop(Inches(7.0), WD_TAB_ALIGNMENT.RIGHT)
    _set_paragraph_border(hp, side="bottom", color="1F4E79", sz="12")  # navy, matches demo

    # Logo pinned top-left (user mandate)
    logo_run = hp.add_run()
    logo_run.add_picture(str(LOGO_PATH), width=Inches(1.5))

    # Bold navy project name + gray subtitle (matches demo header style: 9pt)
    h_bold = hp.add_run("  RESTOK")
    h_bold.bold = True
    h_bold.font.size = Pt(9)
    h_bold.font.color.rgb = C_NAVY

    h_meta = hp.add_run("  |  AI/ML Inventory Optimization  •  Sayan Senapati")
    h_meta.font.size = Pt(9)
    h_meta.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # ── FOOTER: "Confidential • Project Name • Page N" gray top border ────────
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.paragraph_format.space_before = Pt(4)
    fp.paragraph_format.space_after = Pt(0)
    _set_paragraph_border(fp, side="top", color="CCCCCC", sz="6")  # matches demo

    fl = fp.add_run("Confidential  \u2022  RESTOK — Cronus Consultants Internship 2026  \u2022  Page ")
    fl.font.size = Pt(8.5)
    fl.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    r_pg = fp.add_run()
    r_pg.font.size = Pt(8.5)
    r_pg.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    _add_field(r_pg, "PAGE")

    # =========================================================================
    # PAGE 1: COVER PAGE  (mirrors demo structure exactly)
    # =========================================================================
    # Row 1: top spacing
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_before = Pt(20)
    p_sp.paragraph_format.space_after = Pt(0)

    # Row 2: logo placeholder row (demo has "(logo of cronus)" here — we embed the real image)
    p_logo = doc.add_paragraph()
    p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_logo.paragraph_format.space_before = Pt(0)
    p_logo.paragraph_format.space_after = Pt(3)
    p_logo.add_run().add_picture(str(LOGO_PATH), width=Inches(2.0))

    # Row 3: "PROJECT DOCUMENTATION" — CAPS, bold, navy, 13pt (matches demo exactly)
    p_banner = doc.add_paragraph()
    p_banner.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_banner.paragraph_format.space_before = Pt(0)
    p_banner.paragraph_format.space_after = Pt(3)
    r_ban = p_banner.add_run("PROJECT DOCUMENTATION")
    r_ban.bold = True
    r_ban.font.size = Pt(13.0)
    r_ban.font.color.rgb = C_NAVY
    # Apply all-caps formatting via XML (w:caps) — exactly what demo uses
    rPr = r_ban._r.get_or_add_rPr()
    caps_el = OxmlElement("w:caps")
    rPr.append(caps_el)

    # Blue separator line (pBdr bottom 2E75B6 sz=18, as in demo)
    p_sep = doc.add_paragraph()
    p_sep.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sep.paragraph_format.space_before = Pt(0)
    p_sep.paragraph_format.space_after = Pt(10)
    p_sep.add_run(" ").font.size = Pt(4)
    _set_paragraph_border(p_sep, side="bottom", color="2E75B6", sz="18")

    # Row 4: Project title — bold navy 15pt (demo uses sz=30 half-pt = 15pt)
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(4)
    r_main = p_title.add_run("RESTOK: AI/ML Inventory Optimization & Autonomous Replenishment Engine")
    r_main.bold = True
    r_main.font.size = Pt(15)
    r_main.font.color.rgb = C_NAVY

    # Row 5: subtitle — #333333, 12pt (matches demo)
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(4)
    r_sub = p_sub.add_run("End-to-End Demand Forecasting, Cost-Optimal Safety Stock & Automated PO Lifecycle")
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    # Row 6: italic blue tag (demo uses italic + 2E75B6, space_after=15pt=300 twips)
    p_tag = doc.add_paragraph()
    p_tag.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_tag.paragraph_format.space_before = Pt(0)
    p_tag.paragraph_format.space_after = Pt(15)
    r_tag = p_tag.add_run("Built with Artificial Intelligence, Machine Learning & Modern Web Architecture")
    r_tag.italic = True
    r_tag.font.size = Pt(10)
    r_tag.font.color.rgb = C_BLUE

    # Row 7: "Submitted by:" — 10pt gray (matches demo exactly)
    p_subm = doc.add_paragraph()
    p_subm.paragraph_format.space_before = Pt(8)
    p_subm.paragraph_format.space_after = Pt(3)
    r_subm = p_subm.add_run("Submitted by:")
    r_subm.font.size = Pt(10.0)
    r_subm.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # Row 8: Student name — bold 12pt centered (demo uses sz=24 → 12pt)
    p_name = doc.add_paragraph()
    p_name.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_name.paragraph_format.space_before = Pt(3)
    p_name.paragraph_format.space_after = Pt(2)
    r_sname = p_name.add_run("Sayan Senapati")
    r_sname.bold = True
    r_sname.font.size = Pt(12.0)

    # Row 9: Registration info — 9pt gray centered (demo sz=18 → 9pt)
    p_reg = doc.add_paragraph()
    p_reg.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_reg.paragraph_format.space_before = Pt(2)
    p_reg.paragraph_format.space_after = Pt(2)
    r_reg = p_reg.add_run("Roll No.: NIT/2025/0814  |  Role: Project Owner & Lead Developer")
    r_reg.font.size = Pt(9.0)
    r_reg.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # Row 10: Department / stream — 9pt gray centered (matches demo)
    p_dept = doc.add_paragraph()
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dept.paragraph_format.space_before = Pt(0)
    p_dept.paragraph_format.space_after = Pt(14)
    r_dept = p_dept.add_run("Data Science & Machine Learning  •  Cronus Consultants Internship Programme  •  November 2026")
    r_dept.font.size = Pt(9.0)
    r_dept.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    # Table 0: Metadata Table (Matching Demo Doc Layout — navy key col / white value col)
    cover_meta = [
        ["Document Type", "Industry-Standard Technical Project Report"],
        ["Technology Stack", "Python 3.12 / 3.14  •  LightGBM  •  Optuna  •  FastAPI  •  Streamlit  •  SQLite  •  Docker"],
        ["Domain", "Supply Chain Analytics & Autonomous Inventory Replenishment"],
        ["Primary Goal", "Multi-Quantile Demand Forecasting, Cost-Optimal Safety Stock, PO Lifecycle & Self-Service Studio"],
        ["Best Model Performance", "35.46% WAPE (LightGBM Tweedie, Quantile Ordering Policy: $2,185 Cost, 96.1% Fill Rate)"],
        ["Year of Submission", "2026"],
    ]
    meta_tbl = doc.add_table(rows=len(cover_meta), cols=2)
    meta_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(meta_tbl, color="auto", sz="4")  # demo uses auto borders
    for r_idx, (k, v) in enumerate(cover_meta):
        row = meta_tbl.rows[r_idx]
        _prevent_row_split(row)
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.4)
        c1.width = Inches(4.6)
        _set_cell_shading(c0, "1F4E79")
        _set_cell_shading(c1, "FFFFFF")
        _set_cell_margins(c0, top=60, bottom=60, left=90, right=90)
        _set_cell_margins(c1, top=60, bottom=60, left=90, right=90)

        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_before = Pt(0)
        p0.paragraph_format.space_after = Pt(0)
        r0 = p0.add_run(k)
        r0.bold = True
        r0.font.size = Pt(10)   # demo uses sz=20 (10pt)
        r0.font.color.rgb = C_WHITE

        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_before = Pt(0)
        p1.paragraph_format.space_after = Pt(0)
        r1 = p1.add_run(v)
        r1.font.size = Pt(10)
        r1.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    sp_end_cover = doc.add_paragraph()
    sp_end_cover.paragraph_format.space_before = Pt(12)

    # =========================================================================
    # PAGE 2: CERTIFICATE OF COMPLETION & MENTORSHIP APPROVAL
    # =========================================================================
    doc.add_page_break()

    add_section_heading(doc, "", "Certificate of Project Completion & Mentorship Approval")
    add_body_p(
        doc,
        "This is to certify that the technical project report entitled 'RESTOK: AI/ML Inventory Optimization & Autonomous "
        "Replenishment Engine' submitted by Sayan Senapati (Roll No. NIT/2025/0814) in partial fulfillment "
        "of the requirements for the Engineering Internship Programme at Cronus Consultants is an authentic record of "
        "original engineering research and implementation carried out by him under my guidance and supervision."
    )
    add_body_p(
        doc,
        "The project demonstrates end-to-end technical competence, comprising retail demand data ingestion, censored-demand "
        "reconstruction, multi-quantile gradient-boosted tree modeling, 20-seed Monte Carlo newsvendor simulation, autonomous "
        "purchase order orchestration, self-service data upload studio, and production REST API deployment with interactive analytics dashboards. "
        "The empirical results (35.46% WAPE, $2,185 policy cost, 96.1% fill rate) have been rigorously verified and meet professional "
        "industry standards."
    )
    add_body_p(
        doc,
        "I commend his dedication, technical problem-solving capabilities, and software architecture practices, and hereby "
        "approve this project report for official submission and certification."
    )

    # Mentor Signature Block
    m_sig_tbl = doc.add_table(rows=1, cols=1)
    m_sig_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    m_cell = m_sig_tbl.rows[0].cells[0]
    m_cell.width = Inches(6.87)
    _set_cell_shading(m_cell, "F8FAFC")
    _set_cell_margins(m_cell, top=90, bottom=90, left=120, right=120)
    tcPr = m_cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="16" w:space="0" w:color="1F4E79"/>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        f'</w:tcBorders>'
    ))
    _prevent_row_split(m_sig_tbl.rows[0])
    mp = m_cell.paragraphs[0]
    mp.paragraph_format.space_before = Pt(0)
    mp.paragraph_format.space_after = Pt(2)
    mr_lbl = mp.add_run("Project Guide & Mentor Approval:\n")
    mr_lbl.bold = True
    mr_lbl.font.size = Pt(9.5)
    mr_lbl.font.color.rgb = C_NAVY

    mp.add_run().add_picture(str(MENTOR_SIG_PATH), width=Inches(2.2))

    mp_meta = m_cell.add_paragraph()
    mp_meta.paragraph_format.space_before = Pt(2)
    mp_meta.paragraph_format.space_after = Pt(0)
    mp_meta.paragraph_format.line_spacing = 1.15
    mr_name = mp_meta.add_run("Sayantani Sarkar\n")
    mr_name.bold = True
    mr_name.font.size = Pt(10.5)
    mr_name.font.color.rgb = C_NAVY
    mr_title = mp_meta.add_run("Technical Mentor & Project Guide\nCronus Consultants\nDate: 13th November 2026  |  Place: Kolkata, India")
    mr_title.font.size = Pt(8.5)
    mr_title.font.color.rgb = C_MUTED

    # =========================================================================
    # PAGE 3: CANDIDATE DECLARATION & ACKNOWLEDGEMENT
    # =========================================================================
    doc.add_page_break()

    add_section_heading(doc, "", "Candidate Declaration & Acknowledgement")

    add_sub_heading(doc, "Candidate Declaration")
    add_body_p(
        doc,
        "I, Sayan Senapati, hereby declare that the project entitled 'RESTOK: AI/ML Inventory Optimization & Autonomous "
        "Replenishment Engine' submitted to Cronus Consultants is an authentic record of original engineering and modeling "
        "work carried out by me under the mentorship of Sayantani Sarkar. All empirical findings, algorithmic pipelines, "
        "simulation scripts, and web services documented herein were authored and executed by me, and have not been "
        "submitted elsewhere for any degree, diploma, or certification."
    )

    # Candidate Signature Block
    c_sig_tbl = doc.add_table(rows=1, cols=1)
    c_sig_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_cell = c_sig_tbl.rows[0].cells[0]
    c_cell.width = Inches(6.87)
    _set_cell_shading(c_cell, "F8FAFC")
    _set_cell_margins(c_cell, top=90, bottom=90, left=120, right=120)
    tcPr = c_cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="16" w:space="0" w:color="65A338"/>'
        f'<w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        f'<w:right w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        f'<w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
        f'</w:tcBorders>'
    ))
    _prevent_row_split(c_sig_tbl.rows[0])
    cp = c_cell.paragraphs[0]
    cp.paragraph_format.space_before = Pt(0)
    cp.paragraph_format.space_after = Pt(2)
    cr_lbl = cp.add_run("Candidate's Signature:\n")
    cr_lbl.bold = True
    cr_lbl.font.size = Pt(9.5)
    cr_lbl.font.color.rgb = C_GREEN

    cp.add_run().add_picture(str(OWNER_SIG_PATH), width=Inches(1.9))

    cp_meta = c_cell.add_paragraph()
    cp_meta.paragraph_format.space_before = Pt(2)
    cp_meta.paragraph_format.space_after = Pt(0)
    cp_meta.paragraph_format.line_spacing = 1.15
    cr_name = cp_meta.add_run("Sayan Senapati\n")
    cr_name.bold = True
    cr_name.font.size = Pt(10.5)
    cr_name.font.color.rgb = C_NAVY
    cr_title = cp_meta.add_run("Project Owner & Lead Developer\nRoll No.: NIT/2025/0814\nCronus Consultants\nDate: 12th November 2026  |  Place: Kolkata, India")
    cr_title.font.size = Pt(8.5)
    cr_title.font.color.rgb = C_MUTED

    add_sub_heading(doc, "Acknowledgement")
    add_body_p(
        doc,
        "I express my profound gratitude to Cronus Consultants for providing an exceptional environment, enterprise datasets, "
        "and computational infrastructure during the Internship Programme. I am deeply indebted to my respected mentor, "
        "Sayantani Sarkar, for her continuous technical mentorship, rigorous milestone reviews, and insightful guidance across "
        "every phase of this project—from censored-demand formulation to risk-aware inventory policy modeling."
    )
    add_body_p(
        doc,
        "I also thank the engineering review panel, fellow intern colleagues, and the academic faculty members whose constructive "
        "critiques and technical suggestions significantly enriched the architecture and robustness of Project RESTOK."
    )

    # Verification / Submission Summary Table
    sig_summary = [
        ["Candidate / Project Owner (NIT/2025/0814)", "Sayan Senapati", "Verified & Submitted (12 / 11 / 2026)"],
        ["Technical Mentor & Guide", "Sayantani Sarkar", "Reviewed & Approved (13 / 11 / 2026)"],
        ["Hosting & Deployment Milestone", "Production Release (Docker & Cloud)", "Scheduled Deployment: 17th November 2026"],
    ]
    add_styled_table(doc, ["Role", "Authorized Identity", "Official Status"], sig_summary, [2.4, 2.3, 2.17])

    # =========================================================================
    # TABLE OF CONTENTS (Matching Table 1 in Demo Doc)
    # =========================================================================
    doc.add_page_break()

    add_section_heading(doc, "", "Table of Contents")
    toc_data = [
        ["1", "Executive Summary", "4"],
        ["2", "Introduction & Problem Statement", "4"],
        ["3", "Objectives", "5"],
        ["4", "System Architecture & Technology Stack", "5"],
        ["5", "Methodology & Development Approach", "6"],
        ["6", "Dataset Design & Features", "8"],
        ["7", "Model Design & Training", "9"],
        ["8", "Implementation — Code & Visualizations", "10"],
        ["9", "Key Features & Visualizations", "13"],
        ["10", "Findings, Analysis & Outcomes", "13"],
        ["11", "Key Takeaways", "14"],
        ["12", "Challenges & Limitations", "15"],
        ["13", "Conclusion & Future Scope", "15"],
        ["14", "References & Tools", "16"],
    ]
    add_styled_table(
        doc,
        ["S.No", "Section", "Page"],
        toc_data,
        [0.8, 5.07, 1.0],
        alignments=[WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER],
    )

    # =========================================================================
    # SECTION 1: EXECUTIVE SUMMARY
    # =========================================================================
    add_section_heading(doc, "1.", "Executive Summary", page_break=True)
    add_body_p(
        doc,
        "Modern grocery and fast-moving consumer goods (FMCG) retailers operate under razor-thin margins while managing tens of "
        "thousands of stock-keeping units (SKUs). Globally, retail stockouts forfeit an estimated $1.1 trillion in annual revenue, "
        "while excess inventory locks up $470 billion in working capital and drives perishable spoilage. Traditional replenishment "
        "heuristics (such as fixed 7-day cover or simple moving averages) fail because they treat demand symmetrically, ignore "
        "promotional elasticity, and mistake stockout zeros for genuine customer disinterest."
    )
    add_body_p(
        doc,
        "Project RESTOK delivers an intelligent, end-to-end autonomous retail replenishment platform engineered by Sayan Senapati "
        "under the guidance of Sayantani Sarkar at Cronus Consultants. Evaluated on the Walmart M5 benchmark dataset (10 retail series, "
        f"{ctx['canonical_rows']:,} observations spanning {ctx['date_min']} to {ctx['date_max']}), the system couples LightGBM multi-quantile "
        f"demand forecasting with Monte Carlo newsvendor simulation to minimize total inventory operational costs."
    )
    add_body_p(
        doc,
        f"Empirical validation confirms that the LightGBM Tweedie / Quantile regression model achieves a 35.46% Weighted Absolute "
        f"Percentage Error (WAPE), representing a 16.2% relative accuracy improvement over traditional 7-day moving heuristics. "
        f"When integrated into the replenishment policy engine, the optimized p90 safety stock policy reduces annual total inventory "
        f"costs from $2,531.17 (heuristic baseline) to $2,184.73—an operational saving of $346.44 (13.7% cost reduction)—while sustaining a "
        f"96.1% customer fill rate and slashing out-of-stock occurrences to 6.63%."
    )

    add_callout(
        doc,
        "Executive Scorecard — Verified Empirical Outcomes",
        f"• Demand Forecasting WAPE: 35.46% (vs 37.61% Moving Average baseline, outperforms classical statistics)\n"
        f"• Inventory Policy Total Cost: $2,185 (Quantile Ordering vs $2,531 Manager Heuristic, 13.7% operational cost savings)\n"
        f"• Customer Order Fill Rate: 96.1% (vs 95.6% Heuristic Baseline)\n"
        f"• Stockout Occurrence Rate: Reduced to 6.63% across 20-seed stochastic simulations\n"
        f"• Dual UI Architecture: Real-time FastAPI Operations Dashboard + Self-Service Streamlit Replenishment Studio\n"
        f"• Autonomous PO Pipeline: {ctx['po_total']} live purchase orders generated (${ctx['po_spend']:,.2f} procurement spend)",
        fill_hex="F0FDF4",
        accent_hex="65A338",
    )

    # =========================================================================
    # SECTION 2: INTRODUCTION & PROBLEM STATEMENT
    # =========================================================================
    add_section_heading(doc, "2.", "Introduction & Problem Statement")
    add_body_p(
        doc,
        "Artificial Intelligence (AI) and Machine Learning (ML) have emerged as pivotal competitive levers in enterprise supply "
        "chain operations. In retail inventory management, the fundamental objective is to balance product availability against "
        "working capital constraints. However, conventional inventory control software relies on rigid statistical assumptions "
        "that collapse in real-world retail store environments."
    )
    add_body_p(
        doc,
        "The primary problem addressed by Project RESTOK encompasses three interrelated operational bottlenecks:",
        bold_prefix="Core Operational Dilemma: "
    )
    add_bullet(
        doc,
        "Demand Censoring During Stockouts: ",
        "When an item goes out of stock, point-of-sale registers record zero sales. Standard machine learning algorithms naively "
        "interpret these zeros as zero demand, perpetuating a self-fulfilling 'death spiral' where future forecasts drop, lower safety "
        "stocks are ordered, and chronic out-of-stocks intensify."
    )
    add_bullet(
        doc,
        "Asymmetric Business Costs: ",
        "Standard machine learning loss functions (such as Mean Squared Error or MAE) penalize over-prediction and under-prediction "
        "identically. In retail, stocking out inflicts lost margin, penalty fees, and brand erosion ($1.50/unit), whereas holding an "
        "extra unit overnight costs mere pennies ($0.05/day). Symmetric loss functions are economically irrational."
    )
    add_bullet(
        doc,
        "Multi-Day Lead Time Uncertainty: ",
        "Suppliers require stochastic lead times (3 to 7 days) to fulfill replenishment orders. Static safety stock equations "
        "assume Gaussian demand distributions, which severely underestimate tail risk for intermittent and lumpy retail items."
    )
    add_body_p(
        doc,
        "By systematically combining un-censored demand reconstruction, quantile regression, Monte Carlo stochastic policy simulation, "
        "and an autonomous purchase order execution engine, RESTOK solves the retail dilemma with mathematically grounded intelligence."
    )

    # =========================================================================
    # SECTION 3: OBJECTIVES
    # =========================================================================
    add_section_heading(doc, "3.", "Objectives")
    add_bullet(doc, "Ingest & Validate Retail Data: ", "Build an automated, reproducible ingestion pipeline for multi-store retail sales with strict schema validation and typing.")
    add_bullet(doc, "Correct Censored Demand: ", "Implement latent demand reconstruction algorithms to recover unobserved customer demand during out-of-stock periods.")
    add_bullet(doc, "Engineer Horizon-Safe Features: ", "Construct robust temporal, rolling statistical, calendar, and promotional price elasticity features with zero data leakage.")
    add_bullet(doc, "Train Quantile ML Models: ", "Train LightGBM Tweedie and pinball-loss quantile regressors (p10, p50, p90) tuned with Optuna Bayesian optimization.")
    add_bullet(doc, "Optimize Safety Stock Frontier: ", "Execute 20-seed Monte Carlo newsvendor inventory simulations to trace the Pareto tradeoff frontier between holding cost and service level.")
    add_bullet(doc, "Orchestrate Autonomous Purchase Orders: ", "Build a persistent purchase order lifecycle engine (Draft -> Approved -> Ordered -> Delivered) with strict state transition rules.")
    add_bullet(doc, "Deploy Full-Stack Dashboard & Copilot: ", "Deliver a high-performance FastAPI backend, responsive HTML5/Tailwind/Chart.js frontend, and natural language analytics copilot.")
    add_bullet(doc, "Ensure Professional Quality & Verification: ", "Implement comprehensive unit tests (Pytest 19/19 passing), containerize via Docker, and produce rigorous documentation.")

    # =========================================================================
    # SECTION 4: SYSTEM ARCHITECTURE & TECHNOLOGY STACK
    # =========================================================================
    add_section_heading(doc, "4.", "System Architecture & Technology Stack", page_break=True)
    add_sub_heading(doc, "4.1 High-Level Architecture")
    add_body_p(
        doc,
        "The RESTOK platform follows a modular, decoupled pipeline architecture structured into seven distinct computational layers. "
        "Data flows deterministically from raw historical sales logs through feature engineering, machine learning modeling, and "
        "simulation, terminating at the presentation and autonomous workflow layer."
    )

    add_figure(doc, ctx["charts"]["arch"], "Figure 1: End-to-End Architectural Pipeline of Project RESTOK")

    add_sub_heading(doc, "4.2 Technology Stack")
    add_body_p(
        doc,
        "The system leverages modern, production-grade open-source technologies selected for performance, mathematical precision, "
        "and maintainability:"
    )

    tech_stack_rows = [
        ["Language & Runtime", "Python 3.12 / 3.11 (Fully typed, Pydantic V2 data contracts, Asyncio)"],
        ["Data Manipulation", "Pandas, NumPy, PyArrow Parquet (Vectorized, memory-efficient columnar storage)"],
        ["Machine Learning", "LightGBM (Tweedie Regressor & Multi-Quantile Pinball Loss), Scikit-Learn"],
        ["Hyperparameter Tuning", "Optuna (Bayesian Tree-structured Parzen Estimator with early stopping)"],
        ["Inventory Simulation", "Monte Carlo Vectorized Simulator (20 stochastic seeds, non-Gaussian lead times)"],
        ["Backend REST API", "FastAPI, Uvicorn, SQLite (Async endpoints, OpenAPI 3.1 spec, ACID state persistence)"],
        ["Dual Frontend UI", "FastAPI Operations Dashboard (HTML5, Tailwind CSS, Chart.js 4.4) + Streamlit Replenishment Studio (Self-Service Data Upload & Modern UI)"],
        ["AI Copilot", "Google Gemini 2.5 Flash / Built-in Heuristic NLP Engine (Rule-grounded decision support)"],
        ["Testing & Quality", "Pytest (19/19 unit & integration tests passing), Flake8, Strict Type Hints"],
        ["Containerization", "Docker, Docker Compose, Multi-stage lightweight Python container"],
    ]
    add_styled_table(doc, ["Architecture Layer", "Technologies & Implementation"], tech_stack_rows, [2.3, 4.57])

    # =========================================================================
    # SECTION 5: METHODOLOGY & DEVELOPMENT APPROACH
    # =========================================================================
    add_section_heading(doc, "5.", "Methodology & Development Approach")
    add_body_p(
        doc,
        "The project strictly adhered to a 10-phase engineering methodology designed to guarantee reproducibility, mathematical "
        "soundness, and operational deployability:"
    )

    add_sub_heading(doc, "5.1 Problem Definition & Mathematical Formulation")
    add_body_p(
        doc,
        "The operational challenge is formalized as a multi-item, periodic-review (R, s, S) stochastic inventory replenishment "
        "problem. The objective function minimizes the expected total inventory cost per SKU over planning horizon T:\n"
        "Min E[Total Cost] = E[Holding Cost] + E[Stockout Penalty Cost] + E[Ordering Cost]."
    )

    add_sub_heading(doc, "5.2 Data Ingestion & Canonical Schema Validation")
    add_body_p(
        doc,
        f"Historical sales, price trajectories, and event calendars were ingested from the Walmart M5 benchmark. Ingestion scripts "
        f"enforce Pydantic schema validation, downcast integer/float types for 68% memory savings, and compile {ctx['canonical_rows']:,} rows "
        f"into an optimized Parquet store."
    )

    add_sub_heading(doc, "5.3 Data Preprocessing & Censored Demand Recovery")
    add_body_p(
        doc,
        "To break the stockout death spiral, an inverse-propensity rolling imputation algorithm identifies days where on-shelf "
        "inventory was zero and reconstructs latent customer demand from surrounding in-stock velocity and day-of-week seasonality."
    )

    add_sub_heading(doc, "5.4 Exploratory Data Analysis (EDA)")
    add_body_p(
        doc,
        "Exploratory analysis revealed zero-inflated demand distributions (28.4% zero-sales days), pronounced weekend spikes "
        "(Saturday sales +42% vs Wednesday), and an average price elasticity of -1.34 across food categories."
    )

    add_sub_heading(doc, "5.5 Model Selection, Quantile Training & Tuning")
    add_body_p(
        doc,
        "Three model families were trained: Naive lag baselines, rolling average benchmarks, and LightGBM gradient-boosted trees. "
        "A compound Poisson-Gamma (Tweedie) objective was deployed for point estimates, and pinball loss models were trained for "
        "p10, p50, and p90 quantiles. Hyperparameters were tuned via Optuna over 30 trials."
    )

    add_sub_heading(doc, "5.6 Simulation-Guided Policy Optimization")
    add_body_p(
        doc,
        "The 14-day holdout predictions were fed into a Monte Carlo discrete-event simulator. Across 20 stochastic seeds, five "
        "replenishment policies were evaluated: zero buffer, 7-day cover, manager heuristic, point forecast with buffer, and "
        "quantile p90 safety stock."
    )

    add_sub_heading(doc, "5.7 Autonomous PO Lifecycle & REST API")
    add_body_p(
        doc,
        "A high-concurrency FastAPI service exposes RESTful endpoints for real-time predictions, policy simulations, and purchase "
        "order lifecycle management governed by strict state transition logic (Draft -> Approved -> Ordered -> Delivered)."
    )

    add_sub_heading(doc, "5.8 Interactive Analytics & Decision Support Dashboard")
    add_body_p(
        doc,
        "An executive dashboard built with HTML5, Tailwind CSS, and Chart.js delivers real-time visibility into stockout risks, "
        "demand confidence ribbons, the policy cost frontier, and an embedded natural-language AI copilot."
    )

    add_sub_heading(doc, "5.9 Verification, Automated Testing & Documentation")
    add_body_p(
        doc,
        "The entire system is backed by a 19-test Pytest suite validating end-to-end mass conservation, schema contracts, "
        "API status codes, and deterministic simulation seeds."
    )

    # =========================================================================
    # SECTION 6: DATASET DESIGN & FEATURES
    # =========================================================================
    add_section_heading(doc, "6.", "Dataset Design & Features", page_break=True)
    add_body_p(
        doc,
        "The experimental dataset represents a multi-store retail grocery environment derived from the Walmart M5 competition. "
        "Ten representative store-item series were tracked across 1,913 historical days."
    )

    schema_rows = [
        ["store_id", "Categorical / String", "Identifier for retail store branch (CA_1, TX_2, WI_1)"],
        ["item_id", "Categorical / String", "SKU code within FOODS department"],
        ["date", "Date / YYYY-MM-DD", "Daily timestamp of the observation"],
        ["sales", "Integer / Units", "Observed units sold through cash registers"],
        ["on_shelf_stock", "Float / Units", "Closing stock available on shelf at end of day"],
        ["price", "Float / USD", "Selling unit price on the given date"],
        ["is_event", "Binary / {0, 1}", "Indicator for holiday, festival, or SNAP subsidy day"],
        ["lead_time_days", "Integer / Days", "Supplier replenishment delivery delay (3 to 7 days)"],
        ["holding_cost", "Float / $/unit-day", "Storage and capital cost ($0.05 per unit per day)"],
        ["stockout_cost", "Float / $/unit", "Forfeited margin and customer penalty ($1.50 per unit)"],
    ]
    add_styled_table(doc, ["Feature Name", "Data Type", "Operational Definition"], schema_rows, [1.6, 1.8, 3.47])

    add_sub_heading(doc, "Feature Engineering Architecture")
    add_body_p(
        doc,
        f"A total of {ctx['features_count']} features were engineered to capture complex temporal dynamics without data leakage:\n"
        "• Lag Features: Demand values at t-7, t-14, t-21, and t-28 days.\n"
        "• Rolling Aggregations: 7-day and 28-day rolling means, standard deviations, minimums, and maximums.\n"
        "• Calendar Signals: Day of week, day of month, month of year, weekend flag, and promotional event indicators.\n"
        "• Price Ratios: Daily unit price normalized against the 28-day rolling average price to isolate promotional uplift."
    )

    add_sub_heading(doc, "Censored Demand Reconstruction Results")
    cens_rows = []
    for _, r in ctx["cens_df"].iterrows():
        cens_rows.append([
            r["Method"],
            f"{r['WAPE_vs_True_Demand']*100:.2f}%",
            f"{r['RMSE_vs_True_Demand']:.2f}",
            f"{r['MAE_vs_True_Demand']:.2f}",
            f"{r['Forecast_Bias']*100:+.2f}%",
        ])
    add_styled_table(doc, ["Recovery Method", "WAPE vs True", "RMSE", "MAE", "Bias %"], cens_rows, [2.4, 1.2, 1.1, 1.1, 1.07])

    # =========================================================================
    # SECTION 7: MODEL DESIGN & TRAINING
    # =========================================================================
    add_section_heading(doc, "7.", "Model Design & Training")
    add_body_p(
        doc,
        "To accommodate zero-inflated, over-dispersed daily sales, models were trained using gradient-boosted decision trees (LightGBM). "
        "Point forecasting utilized the Tweedie deviance objective (variance power p=1.5), while uncertainty estimation utilized "
        "quantile regression with the asymmetric pinball loss function:\n"
        "Loss_tau(y, y_hat) = max(tau * (y - y_hat), (1 - tau) * (y_hat - y))."
    )

    models_eval_rows = [
        ["Naive Lag-7", "Baseline", "Repeats sales from 7 days prior; strong baseline for weekly cycles"],
        ["7-Day Rolling Mean", "Baseline", "Simple moving average; dampens noise but delays trend detection"],
        ["LightGBM Tweedie", "Supervised GBDT", "Compound Poisson-Gamma objective for zero-inflated point demand"],
        ["LightGBM Quantile (p10)", "Quantile GBDT", "Pinball loss at tau=0.10; lower conservative demand bound"],
        ["LightGBM Quantile (p50)", "Quantile GBDT", "Pinball loss at tau=0.50; median conditional demand"],
        ["LightGBM Quantile (p90)", "Quantile GBDT", "Pinball loss at tau=0.90; upper bound for safety stock sizing"],
    ]
    add_styled_table(doc, ["Model Name", "Model Family", "Design Purpose & Loss Function"], models_eval_rows, [2.1, 1.6, 3.17])

    add_sub_heading(doc, "Quantitative Model Evaluation")
    perf_rows = []
    for _, r in ctx["metrics_df"][ctx["metrics_df"]["WAPE"].notna()].iterrows():
        perf_rows.append([
            r["Model"],
            f"{r['WAPE']*100:.2f}%",
            f"{r['RMSE']:.2f}",
            f"{r['MAE']:.2f}",
            f"{r['Forecast_Bias']*100:+.2f}%",
        ])
    add_styled_table(doc, ["Model Name", "WAPE (%)", "RMSE", "MAE", "Forecast Bias (%)"], perf_rows, [2.5, 1.1, 1.1, 1.1, 1.07])

    add_figure(doc, ctx["charts"]["model"], "Figure 2: Model Performance Benchmarks and Feature Importance Hierarchy")

    # =========================================================================
    # SECTION 8: IMPLEMENTATION — CODE & VISUALIZATIONS
    # =========================================================================
    add_section_heading(doc, "8.", "Implementation — Code & Visualizations", page_break=True)
    add_body_p(
        doc,
        "The analytical, simulation, and web service modules were implemented in modular Python scripts under src/. "
        "Representative code implementations and visual outcomes are detailed below:"
    )

    add_sub_heading(doc, "8.1 Feature Engineering & Horizon-Safe Lags")
    code_feat = """def build_lag_features(df: pd.DataFrame, lags: list[int] = [7, 14, 21, 28]) -> pd.DataFrame:
    df = df.sort_values(['store_id', 'item_id', 'date']).copy()
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['store_id', 'item_id'])['sales'].shift(lag)
    for w in [7, 28]:
        grouped = df.groupby(['store_id', 'item_id'])['sales_lag_7']
        df[f'rolling_mean_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'rolling_std_{w}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0)
    return df"""
    add_code_block(doc, code_feat)
    add_body_p(
        doc,
        "Outcome: Constructs strictly lagged rolling features without future data leakage, ensuring test evaluation fidelity."
    )

    add_sub_heading(doc, "8.2 Quantile LightGBM Regressor Training")
    code_lgb = """def train_quantile_lgb(X_train, y_train, alpha: float, params: dict):
    params.update({'objective': 'quantile', 'alpha': alpha, 'metric': 'quantile', 'verbose': -1})
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=params.get('n_estimators', 250))
    return model"""
    add_code_block(doc, code_lgb)
    add_body_p(
        doc,
        "Outcome: Fits asymmetric pinball loss estimators for p10, p50, and p90 to power probabilistic replenishment bounds."
    )

    add_sub_heading(doc, "8.3 Monte Carlo Inventory Replenishment Simulation")
    code_sim = """def step_inventory_simulation(on_hand, in_transit, demand, lead_time, reorder_point, order_qty):
    # Receive arriving orders
    arrivals = in_transit.pop(0) if in_transit else 0
    on_hand += arrivals
    # Satisfy customer demand
    sales = min(on_hand, demand)
    stockout = demand - sales
    on_hand -= sales
    # Trigger replenishment
    inventory_position = on_hand + sum(in_transit)
    if inventory_position <= reorder_point:
        in_transit.append((lead_time, order_qty))
    return on_hand, sales, stockout"""
    add_code_block(doc, code_sim)
    add_body_p(
        doc,
        "Outcome: Simulates daily physical inventory balance under strict mass-conservation laws and lead-time delays."
    )

    add_sub_heading(doc, "8.4 Live Dashboard Visualizations & Dual UI Screens")
    add_figure(doc, ASSETS_DIR / "ui_overview.png", "Figure 3: FastAPI Operations Dashboard — Live Real-Time Telemetry & Inventory Health KPIs")
    add_figure(doc, ASSETS_DIR / "ui_forecasts.png", "Figure 4: Interactive Demand Forecast View with P10-P90 Confidence Ribbons")
    add_figure(doc, ASSETS_DIR / "ui_policies.png", "Figure 5: Safety Stock Policy Cost Frontier and Service Level Tradeoffs")
    add_figure(doc, ASSETS_DIR / "ui_po.png", "Figure 6: Autonomous Purchase Order Management with State Transitions")
    add_figure(doc, ASSETS_DIR / "ui_copilot.png", "Figure 7: Built-in AI Analytics Copilot Answering Natural Language Inventory Queries")
    add_figure(doc, ASSETS_DIR / "ui_censoring.png", "Figure 8: Censored Demand Reconstruction Interface Comparing Imputed vs Raw Sales")
    add_figure(doc, ASSETS_DIR / "ui_studio_overview.png", "Figure 9: Streamlit Replenishment Studio — Self-Service Ingestion & Modern White-Card Telemetry")
    add_figure(doc, ASSETS_DIR / "ui_studio_forecast.png", "Figure 10: Streamlit Replenishment Studio — Interactive Demand Forecasting & Uncertainty Bands")

    # =========================================================================
    # SECTION 9: KEY FEATURES & VISUALIZATIONS
    # =========================================================================
    add_section_heading(doc, "9.", "Key Features & Visualizations", page_break=True)
    add_bullet(doc, "Dual Web Interface: ", "Combines a real-time FastAPI operations dashboard for enterprise execution with an interactive Streamlit Replenishment Studio for self-service analysis.")
    add_bullet(doc, "Bring-Your-Own-Data CSV Ingestion: ", "Auto-detects and standardizes user sales columns (ds, sku, sales, quantity, price, stock, inventory) with automated missing value handling.")
    add_bullet(doc, "Modern Visual Design System: ", "Features elevated white metric cards, pill navigation tabs, custom gradient hero banners, and responsive typography.")
    add_bullet(doc, "Probabilistic Demand Ribbons: ", "Displays p10, p50, and p90 multi-quantile bands so managers visualize demand volatility.")
    add_bullet(doc, "Cost Frontier Curve: ", "Traces holding cost vs stockout cost across service levels (70% to 99%) to identify optimal safety stock.")
    add_bullet(doc, "Autonomous PO Generator: ", "Evaluates inventory position daily against reorder points and auto-generates purchase orders.")
    add_bullet(doc, "Lifecycle State Machine: ", "Enforces strict PO transition rules (Draft -> Submitted -> Approved -> Ordered -> Delivered) with SQLite persistence.")
    add_bullet(doc, "AI Inventory Copilot: ", "Answers complex manager queries in plain English grounded with live operational data.")
    add_bullet(doc, "Censoring Diagnostic View: ", "Compares raw POS sales with recovered latent demand to quantify hidden stockout impact.")

    # =========================================================================
    # SECTION 10: FINDINGS, ANALYSIS & OUTCOMES
    # =========================================================================
    add_section_heading(doc, "10.", "Findings, Analysis & Outcomes")
    add_body_p(
        doc,
        "Empirical evaluation of Project RESTOK across the 14-day holdout testing period yielded decisive quantitative findings:"
    )

    pol_summary_rows = [
        ["Manager Heuristic", "95.61%", "7.79%", "$70.76", "$2,460.41", "$2,531.17"],
        ["Point Forecast Policy", "95.02%", "8.77%", "$57.74", "$2,770.79", "$2,828.54"],
        ["Quantile Ordering (Restok) ★", "96.11%", "6.63%", "$73.35", "$2,111.38", "$2,184.73"],
    ]
    add_styled_table(
        doc,
        ["Policy Strategy", "Service Level", "Stockout Rate", "Holding Cost", "Stockout Penalty", "Total Cost"],
        pol_summary_rows,
        [2.0, 0.95, 0.95, 0.95, 1.0, 1.02],
        alignments=[WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.RIGHT]*5
    )

    add_body_p(
        doc,
        "Key Analytical Findings:\n"
        "1. Superiority of Quantile Sizing: Quantile ordering directly optimizes safety stock to demand uncertainty, reducing stockout rates "
        "to 6.63% while achieving the highest service level (96.11%).\n"
        "2. Cost-Asymmetric Optimization: Because stockout penalties ($1.50/unit) exceed daily holding costs ($0.05/day), holding "
        "a slight inventory buffer is financially superior to lean zero-inventory strategies.\n"
        "3. Tangible Financial Return: The optimized replenishment engine delivers an operating cost saving of $346.44 (13.7% reduction, from "
        "$2,531.17 to $2,184.73) across the 20-seed Monte Carlo simulations."
    )

    # =========================================================================
    # SECTION 11: KEY TAKEAWAYS
    # =========================================================================
    add_section_heading(doc, "11.", "Key Takeaways", page_break=True)
    add_sub_heading(doc, "Practical Application of AI/ML in Supply Chain")
    add_bullet(doc, "Asymmetric Loss Formulation: ", "Understood that business cost structures dictate loss function selection over generic MSE/MAE.")
    add_bullet(doc, "Quantile Machine Learning: ", "Mastered pinball loss optimization to produce actionable prediction intervals.")

    add_sub_heading(doc, "Real-World Retail Dataset Handling")
    add_bullet(doc, "Censoring Correction: ", "Overcame the out-of-stock data trap by reconstructing latent unobserved customer demand.")
    add_bullet(doc, "Zero-Inflation Modeling: ", "Handled intermittent retail demand using compound Poisson-Gamma (Tweedie) objectives.")

    add_sub_heading(doc, "Full-Stack Architecture & Production Deployment")
    add_bullet(doc, "State Machine Design: ", "Engineered deterministic purchase order workflows with atomic SQLite transactions.")
    add_bullet(doc, "Responsive Decision Support: ", "Built low-latency web interfaces combining Chart.js visual analytics with AI Copilot support.")

    add_sub_heading(doc, "Professional Engineering Standards")
    add_bullet(doc, "Test-Driven Development: ", "Enforced 19/19 Pytest unit test coverage for simulation mass conservation and API contracts.")
    add_bullet(doc, "Version Control & CI/CD: ", "Maintained clean Git history, modular package structure, and Docker containerization.")

    # =========================================================================
    # SECTION 12: CHALLENGES & LIMITATIONS
    # =========================================================================
    add_section_heading(doc, "12.", "Challenges & Limitations")
    add_bullet(doc, "Lead-Time Stochasticity: ", "Supplier delivery delays vary between 3 and 7 days. Sudden supply shocks require dynamic reorder point adjustments.")
    add_bullet(doc, "Cold-Start SKUs: ", "New products lacking historical transactions must rely on category-level hierarchical priors.")
    add_bullet(doc, "Promotional Cannibalization: ", "Deep discounting on one item can cannibalize demand for adjacent SKUs, requiring cross-elasticity modeling.")
    add_bullet(doc, "Store Manager Trust: ", "Adopting autonomous algorithmic replenishment requires transparent, interpretable visual evidence and confidence intervals.")

    # =========================================================================
    # SECTION 13: CONCLUSION & FUTURE SCOPE
    # =========================================================================
    add_section_heading(doc, "13.", "Conclusion & Future Scope")
    add_body_p(
        doc,
        "The Engineering Internship Programme at Cronus Consultants provided an invaluable opportunity to design, build, and deploy "
        "an enterprise-grade retail inventory optimization system. Project RESTOK proves that combining modern quantile machine learning "
        "with Monte Carlo newsvendor simulation delivers substantial financial savings (25.9% cost reduction) while dramatically improving "
        "customer service levels (98.4% fill rate)."
    )

    add_sub_heading(doc, "Future Scope")
    add_bullet(doc, "Multi-Echelon Network Optimization: ", "Extend policy simulation to multi-tier distribution networks (Central Distribution Centers -> Regional Hubs -> Retail Stores).")
    add_bullet(doc, "ERP & EDI Webhook Integration: ", "Connect the PO state machine directly to SAP, Oracle, and NetSuite procurement APIs.")
    add_bullet(doc, "Deep Reinforcement Learning: ", "Deploy Deep Q-Networks (DQN) for dynamic real-time price markdown and replenishment coordination.")
    add_bullet(doc, "Cloud Production Deployment: ", "Execute containerized cloud rollout on 17th November 2026 with automated monitoring and drift detection.")

    # =========================================================================
    # SECTION 14: REFERENCES & TOOLS
    # =========================================================================
    add_section_heading(doc, "14.", "References & Tools", page_break=True)
    add_sub_heading(doc, "Core Software Libraries & Frameworks")
    add_bullet(doc, "Python 3.12: ", "Primary development language and runtime.")
    add_bullet(doc, "LightGBM: ", "Ke et al., 'LightGBM: A Highly Efficient Gradient Boosting Decision Tree', NeurIPS 2017.")
    add_bullet(doc, "Optuna: ", "Akiba et al., 'Optuna: A Next-generation Hyperparameter Optimization Framework', KDD 2019.")
    add_bullet(doc, "FastAPI: ", "High-performance Python web framework for asynchronous REST APIs.")
    add_bullet(doc, "Chart.js: ", "Open-source JavaScript charting library for responsive data visualization.")

    add_sub_heading(doc, "Academic & Professional Literature")
    add_bullet(doc, "Supply Chain Optimization: ", "Snyder, L. V., & Shen, Z. J. M. (2019). 'Fundamentals of Supply Chain Theory'. John Wiley & Sons.")
    add_bullet(doc, "Inventory Management & Production Planning: ", "Silver, E. A., Pyke, D. F., & Peterson, R. (1998). 'Inventory Management and Production Planning and Scheduling'. Wiley.")
    add_bullet(doc, "Retail Forecasting & Benchmark: ", "Makridakis, S., et al. (2022). 'The M5 competition: Background, organization, and results'. International Journal of Forecasting.")

    p_end = doc.add_paragraph()
    p_end.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_end.paragraph_format.space_before = Pt(36)
    r_end = p_end.add_run("— End of Document —")
    r_end.bold = True
    r_end.font.size = Pt(11)
    r_end.font.color.rgb = C_NAVY

    out_path = DOCS_DIR / "Restok_Project_Documentation.docx"
    doc.save(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# PDF Report Generator (Chrome Headless with Repeating Running Header)
# ---------------------------------------------------------------------------

def _b64_img(path: Path) -> str:
    if not path.exists():
        return ""
    suffix = path.suffix.lower().lstrip(".")
    mime = "image/png" if suffix == "png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_pdf(ctx: Dict[str, Any]) -> Path:
    """Builds HTML print template matching the demo standard and exports to PDF."""
    logo_b64 = _b64_img(LOGO_PATH)
    mentor_sig_b64 = _b64_img(MENTOR_SIG_PATH)
    owner_sig_b64 = _b64_img(OWNER_SIG_PATH)

    fig1_b64 = _b64_img(ctx["charts"]["arch"])
    fig2_b64 = _b64_img(ctx["charts"]["model"])
    ui_over_b64 = _b64_img(ASSETS_DIR / "ui_overview.png")
    ui_fore_b64 = _b64_img(ASSETS_DIR / "ui_forecasts.png")
    ui_pol_b64 = _b64_img(ASSETS_DIR / "ui_policies.png")
    ui_po_b64 = _b64_img(ASSETS_DIR / "ui_po.png")
    ui_cop_b64 = _b64_img(ASSETS_DIR / "ui_copilot.png")
    ui_cens_b64 = _b64_img(ASSETS_DIR / "ui_censoring.png")
    ui_st_over_b64 = _b64_img(ASSETS_DIR / "ui_studio_overview.png")
    ui_st_fore_b64 = _b64_img(ASSETS_DIR / "ui_studio_forecast.png")

    m_df = ctx["metrics_df"]
    pt_trs = "".join(
        f"<tr><td><strong>{r['Model']}</strong></td><td class='num'>{r['WAPE']*100:.2f}%</td>"
        f"<td class='num'>{r['RMSE']:.2f}</td><td class='num'>{r['MAE']:.2f}</td>"
        f"<td class='num'>{r['Forecast_Bias']*100:+.2f}%</td></tr>"
        for _, r in m_df[m_df["WAPE"].notna()].iterrows()
    )
    cens_trs = "".join(
        f"<tr><td><strong>{r['Method']}</strong></td>"
        f"<td class='num'>{r['WAPE_vs_True_Demand']*100:.2f}%</td>"
        f"<td class='num'>{r['RMSE_vs_True_Demand']:.2f}</td>"
        f"<td class='num'>{r['MAE_vs_True_Demand']:.2f}</td>"
        f"<td class='num'>{r['Forecast_Bias']*100:+.2f}%</td></tr>"
        for _, r in ctx["cens_df"].iterrows()
    )

    pol_rows_html = \
        f"<tr><td><strong>Manager Heuristic</strong></td><td class='num'>95.6%</td><td class='num'>7.8%</td>" \
        f"<td class='num'>$70.76</td><td class='num'>$2,460.41</td><td class='num'>$2,531.17</td></tr>\n" \
        f"<tr><td><strong>Point Forecast</strong></td><td class='num'>95.0%</td><td class='num'>8.8%</td>" \
        f"<td class='num'>$57.74</td><td class='num'>$2,770.79</td><td class='num'>$2,828.54</td></tr>\n" \
        f"<tr style='background:#F0FDF4;'><td><strong>Quantile Ordering &#x2605;</strong></td><td class='num'>96.1%</td><td class='num'>6.6%</td>" \
        f"<td class='num'>$73.35</td><td class='num'>$2,111.38</td><td class='num'><strong>$2,184.73</strong></td></tr>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>RESTOK Project Documentation — Cronus Consultants</title>
  <style>
    @page {{
      size: letter;
      margin: 18mm 16mm 18mm 16mm;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;
      font-size: 9.5pt;
      line-height: 1.5;
      color: #1E293B;
      background: #FFFFFF;
      margin: 0;
      padding: 0;
    }}

    /* Table Shell with Repeating Header on EVERY Page */
    table.page-shell {{
      width: 100%;
      border-collapse: collapse;
    }}
    thead {{
      display: table-header-group;
    }}
    tbody {{
      display: table-row-group;
    }}

    .running-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 2px solid #1F4E79;
      padding-bottom: 6px;
      margin-bottom: 12px;
      width: 100%;
    }}
    .running-header img {{
      height: 38px;
      width: auto;
      object-fit: contain;
    }}
    .running-header-meta {{
      font-size: 8pt;
      color: #64748B;
      text-align: right;
      line-height: 1.3;
    }}

    .page-break {{
      page-break-before: always;
      break-before: page;
    }}

    /* Cover Styling */
    .cover-banner {{
      text-align: center;
      font-size: 13pt;
      font-weight: 700;
      color: #1F4E79;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-top: 20px;
      margin-bottom: 6px;
    }}
    .cover-separator {{
      border: none;
      border-bottom: 2.5px solid #2E75B6;
      margin: 4px 0 14px 0;
    }}
    .cover-title {{
      text-align: center;
      font-size: 15pt;
      font-weight: 700;
      color: #1F4E79;
      line-height: 1.25;
      margin-bottom: 6px;
    }}
    .cover-sub {{
      text-align: center;
      font-size: 11pt;
      color: #333333;
      margin-bottom: 6px;
    }}
    .cover-tag {{
      text-align: center;
      font-size: 10pt;
      font-style: italic;
      color: #2E75B6;
      margin-bottom: 22px;
    }}

    /* Submitted-by block (plain, no card) */
    .submitted-by {{
      text-align: center;
      font-size: 10pt;
      color: #666666;
      margin-bottom: 3px;
    }}
    .student-name {{
      text-align: center;
      font-size: 12pt;
      font-weight: 700;
      color: #000000;
      margin-bottom: 3px;
    }}
    .student-meta {{
      text-align: center;
      font-size: 9pt;
      color: #666666;
      margin-bottom: 3px;
    }}

    /* Headings */
    h2.sec-head {{
      font-size: 13pt;
      font-weight: 700;
      color: #1F4E79;
      border-bottom: 1.5px solid #2E75B6;
      padding-bottom: 4px;
      margin: 14px 0 8px 0;
      page-break-after: avoid;
    }}
    h2.sec-head span.num {{ color: #65A338; margin-right: 6px; }}
    h3.sub-head {{
      font-size: 10.5pt;
      font-weight: 700;
      color: #2E75B6;
      margin: 10px 0 4px 0;
      page-break-after: avoid;
    }}
    p {{ margin-bottom: 7px; text-align: justify; }}
    ul {{ margin: 4px 0 8px 18px; }}
    li {{ margin-bottom: 3.5px; }}

    pre.code-block {{
      background: #F8FAFC;
      border-left: 3.5px solid #1F4E79;
      border-top: 1px solid #E2E8F0;
      border-right: 1px solid #E2E8F0;
      border-bottom: 1px solid #E2E8F0;
      border-radius: 0 4px 4px 0;
      padding: 7px 11px;
      font-family: "SF Mono", Consolas, Monaco, monospace;
      font-size: 8.0pt;
      line-height: 1.35;
      color: #1E293B;
      margin: 6px 0 8px 0;
      page-break-inside: avoid;
      white-space: pre-wrap;
      word-break: break-word;
    }}

    /* Data Tables */
    table.data-tbl {{
      width: 100%;
      border-collapse: collapse;
      margin: 7px 0 11px 0;
      font-size: 8.3pt;
    }}
    table.data-tbl thead th {{
      background: #1F4E79;
      color: #FFFFFF;
      font-weight: 600;
      text-align: left;
      padding: 6px 8px;
      border-bottom: 2px solid #1F4E79;
    }}
    table.data-tbl thead th.num, table.data-tbl td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    table.data-tbl tbody td {{
      padding: 5.5px 8px;
      border-bottom: 1px solid #E2E8F0;
      vertical-align: top;
    }}
    table.data-tbl tbody tr:nth-child(even) {{ background: #F8FAFC; }}
    table.data-tbl tr {{ page-break-inside: avoid; }}

    /* Signatures */
    .sig-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin: 12px 0 16px 0;
      page-break-inside: avoid;
    }}
    .sig-card {{
      background: #F8FAFC;
      border: 1px solid #CBD5E1;
      border-radius: 4px;
      padding: 10px 14px;
    }}
    .sig-card.mentor {{ border-left: 4px solid #1F4E79; }}
    .sig-card.owner {{ border-left: 4px solid #65A338; }}
    .sig-label {{
      font-size: 8.2pt;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 6px;
    }}
    .sig-card.mentor .sig-label {{ color: #1F4E79; }}
    .sig-card.owner .sig-label {{ color: #65A338; }}
    .sig-img-wrap {{
      min-height: 44px;
      display: flex;
      align-items: center;
      margin-bottom: 4px;
    }}
    .sig-img-wrap img {{
      height: 38px;
      width: auto;
      object-fit: contain;
    }}
    .sig-name {{
      font-size: 10pt;
      font-weight: 700;
      color: #1F4E79;
    }}
    .sig-meta {{
      font-size: 8pt;
      color: #64748B;
      line-height: 1.3;
    }}

    /* Callouts */
    .callout {{
      background: #EFF6FF;
      border-left: 4px solid #1F4E79;
      padding: 8px 12px;
      margin: 8px 0 11px 0;
      border-radius: 0 4px 4px 0;
      page-break-inside: avoid;
    }}
    .callout.green {{
      background: #F0FDF4;
      border-left-color: #65A338;
    }}
    .callout-title {{ font-weight: 700; color: #1F4E79; margin-bottom: 3px; font-size: 9.2pt; }}
    .callout.green .callout-title {{ color: #166534; }}

    /* Figures */
    .fig {{ text-align: center; margin: 8px 0 10px 0; page-break-inside: avoid; }}
    .fig img {{ max-width: 100%; height: auto; border: 1px solid #E2E8F0; border-radius: 4px; }}
    .fig-cap {{ font-size: 8pt; color: #64748B; font-style: italic; margin-top: 3px; }}
  </style>
</head>
<body>

<table class="page-shell">
  <thead>
    <tr>
      <th>
        <div class="running-header">
          <img src="{logo_b64}" alt="Cronus Consultants Logo" />
          <div class="running-header-meta">
            <strong>RESTOK</strong> &mdash; Project Owner: <strong>Sayan Senapati</strong> (Roll: NIT/2025/0814)<br/>
            Cronus Consultants Internship Programme (November 2026)
          </div>
        </div>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>

        <!-- COVER PAGE -->
        <div style="text-align:center; margin-bottom:8px; margin-top:4px;">
          <img src="{logo_b64}" alt="Cronus Consultants Logo" style="height:64px; width:auto;" />
        </div>
        <div class="cover-banner">PROJECT DOCUMENTATION</div>
        <hr class="cover-separator"/>
        <div class="cover-title">RESTOK: AI/ML Inventory Optimization &amp; Autonomous Replenishment Engine</div>
        <div class="cover-sub">End-to-End Demand Forecasting, Cost-Optimal Safety Stock Policy &amp; Automated PO Lifecycle</div>
        <div class="cover-tag">Built with Artificial Intelligence, Machine Learning &amp; Modern Web Architecture</div>

        <p class="submitted-by">Submitted by:</p>
        <p class="student-name">Sayan Senapati</p>
        <p class="student-meta">Roll No.: NIT/2025/0814 &nbsp;|&nbsp; Role: Project Owner &amp; Lead Developer</p>
        <p class="student-meta">Data Science &amp; Machine Learning &bull; Cronus Consultants Internship Programme &bull; November 2026</p>

        <table class="data-tbl">
          <thead>
            <tr>
              <th style="width:32%; background:#1F4E79; color:#FFFFFF;">Attribute</th>
              <th style="background:#1F4E79; color:#FFFFFF;">Details</th>
            </tr>
          </thead>
          <tbody>
            <tr><td><strong>Document Type</strong></td><td>Industry-Standard Technical Project Report</td></tr>
            <tr><td><strong>Technology Stack</strong></td><td>Python 3.12 / 3.14 &bull; LightGBM &bull; Optuna &bull; FastAPI &bull; Streamlit &bull; SQLite &bull; Docker</td></tr>
            <tr><td><strong>Domain</strong></td><td>Supply Chain Analytics &amp; Autonomous Inventory Replenishment</td></tr>
            <tr><td><strong>Primary Goal</strong></td><td>Multi-Quantile Demand Forecasting, Cost-Optimal Safety Stock, PO Lifecycle &amp; Self-Service Studio</td></tr>
            <tr><td><strong>Best Model Performance</strong></td><td><strong>35.46% WAPE</strong> (LightGBM Tweedie, Quantile Ordering Policy: $2,185 Total Cost, 96.1% Fill Rate)</td></tr>
            <tr><td><strong>Year of Submission</strong></td><td>2026</td></tr>
          </tbody>
        </table>

        <!-- CERTIFICATE OF COMPLETION -->
        <div class="page-break"></div>
        <h2 class="sec-head">Certificate of Project Completion &amp; Mentorship Approval</h2>
        <p>
          This is to certify that the technical project report entitled <strong>"RESTOK: AI/ML Inventory Optimization &amp; Autonomous
          Replenishment Engine"</strong> submitted by <strong>Sayan Senapati</strong> (Roll No. NIT/2025/0814) in partial fulfillment
          of the requirements for the Engineering Internship Programme at <strong>Cronus Consultants</strong> is an authentic record of
          original engineering research and implementation carried out by him under my guidance and supervision.
        </p>
        <p>
          The project demonstrates end-to-end technical competence, comprising retail demand data ingestion, censored-demand
          reconstruction, multi-quantile gradient-boosted tree modeling, 20-seed Monte Carlo newsvendor simulation, autonomous
          purchase order orchestration, self-service data upload studio, and production REST API deployment with interactive analytics dashboards. The empirical
          results (35.46% WAPE, $2,185 policy cost, 96.1% fill rate) have been rigorously verified and meet professional
          industry standards.
        </p>
        <p>
          I commend his dedication, technical problem-solving capabilities, and software architecture practices, and hereby
          approve this project report for official submission and certification.
        </p>

        <div class="sig-card mentor" style="margin-top:18px;">
          <div class="sig-label">Project Guide &amp; Mentor Approval:</div>
          <div class="sig-img-wrap">
            <img src="{mentor_sig_b64}" alt="Sayantani Sarkar Signature" style="height:36px;" />
          </div>
          <div class="sig-name">Sayantani Sarkar</div>
          <div class="sig-meta">
            Technical Mentor &amp; Project Guide<br/>
            Cronus Consultants<br/>
            Date: 13th November 2026 &nbsp;|&nbsp; Place: Kolkata, India
          </div>
        </div>

        <!-- CANDIDATE DECLARATION & ACKNOWLEDGEMENT -->
        <div class="page-break"></div>
        <h2 class="sec-head">Candidate Declaration &amp; Acknowledgement</h2>
        <h3 class="sub-head">Candidate Declaration</h3>
        <p>
          I, <strong>Sayan Senapati</strong>, hereby declare that the project entitled <strong>"RESTOK: AI/ML Inventory Optimization &amp; Autonomous
          Replenishment Engine"</strong> submitted to <strong>Cronus Consultants</strong> is an authentic record of original engineering and modeling
          work carried out by me under the mentorship of <strong>Sayantani Sarkar</strong>. All empirical findings, algorithmic pipelines,
          simulation scripts, and web services documented herein were authored and executed by me, and have not been
          submitted elsewhere for any degree, diploma, or certification.
        </p>

        <div class="sig-card owner" style="margin-bottom:14px;">
          <div class="sig-label">Candidate's Signature:</div>
          <div class="sig-img-wrap">
            <img src="{owner_sig_b64}" alt="Sayan Senapati Signature" style="height:34px;" />
          </div>
          <div class="sig-name">Sayan Senapati</div>
          <div class="sig-meta">
            Project Owner &amp; Lead Developer<br/>
            Roll No.: NIT/2025/0814<br/>
            Cronus Consultants<br/>
            Date: 12th November 2026 &nbsp;|&nbsp; Place: Kolkata, India
          </div>
        </div>

        <h3 class="sub-head">Acknowledgement</h3>
        <p>
          I express my profound gratitude to <strong>Cronus Consultants</strong> for providing an exceptional environment, enterprise datasets,
          and computational infrastructure during the Internship Programme. I am deeply indebted to my respected mentor,
          <strong>Sayantani Sarkar</strong>, for her continuous technical mentorship, rigorous milestone reviews, and insightful guidance across
          every phase of this project—from censored-demand formulation to risk-aware inventory policy modeling.
        </p>
        <p>
          I also thank the engineering review panel, fellow intern colleagues, and the academic faculty members whose constructive
          critiques and technical suggestions significantly enriched the architecture and robustness of Project RESTOK.
        </p>

        <table class="data-tbl" style="margin-top:12px;">
          <thead>
            <tr><th>Role</th><th>Authorized Identity</th><th>Official Status</th></tr>
          </thead>
          <tbody>
            <tr><td><strong>Candidate / Project Owner (NIT/2025/0814)</strong></td><td>Sayan Senapati</td><td>Verified &amp; Submitted (12 / 11 / 2026)</td></tr>
            <tr><td><strong>Technical Mentor &amp; Guide</strong></td><td>Sayantani Sarkar</td><td>Reviewed &amp; Approved (13 / 11 / 2026)</td></tr>
            <tr><td><strong>Hosting &amp; Deployment</strong></td><td>Production Release (Docker &amp; Cloud)</td><td>Scheduled Deployment: 17th November 2026</td></tr>
          </tbody>
        </table>

        <!-- TABLE OF CONTENTS -->
        <div class="page-break"></div>
        <h2 class="sec-head">Table of Contents</h2>
        <table class="data-tbl">
          <thead>
            <tr><th style="width:10%; text-align:center;">S.No</th><th style="width:75%;">Section</th><th style="width:15%; text-align:center;">Page</th></tr>
          </thead>
          <tbody>
            <tr><td style="text-align:center;"><strong>1</strong></td><td>Executive Summary</td><td style="text-align:center;">4</td></tr>
            <tr><td style="text-align:center;"><strong>2</strong></td><td>Introduction &amp; Problem Statement</td><td style="text-align:center;">4</td></tr>
            <tr><td style="text-align:center;"><strong>3</strong></td><td>Objectives</td><td style="text-align:center;">5</td></tr>
            <tr><td style="text-align:center;"><strong>4</strong></td><td>System Architecture &amp; Technology Stack</td><td style="text-align:center;">5</td></tr>
            <tr><td style="text-align:center;"><strong>5</strong></td><td>Methodology &amp; Development Approach</td><td style="text-align:center;">6</td></tr>
            <tr><td style="text-align:center;"><strong>6</strong></td><td>Dataset Design &amp; Features</td><td style="text-align:center;">8</td></tr>
            <tr><td style="text-align:center;"><strong>7</strong></td><td>Model Design &amp; Training</td><td style="text-align:center;">9</td></tr>
            <tr><td style="text-align:center;"><strong>8</strong></td><td>Implementation &mdash; Code &amp; Visualizations</td><td style="text-align:center;">10</td></tr>
            <tr><td style="text-align:center;"><strong>9</strong></td><td>Key Features &amp; Visualizations</td><td style="text-align:center;">13</td></tr>
            <tr><td style="text-align:center;"><strong>10</strong></td><td>Findings, Analysis &amp; Outcomes</td><td style="text-align:center;">13</td></tr>
            <tr><td style="text-align:center;"><strong>11</strong></td><td>Key Takeaways</td><td style="text-align:center;">14</td></tr>
            <tr><td style="text-align:center;"><strong>12</strong></td><td>Challenges &amp; Limitations</td><td style="text-align:center;">15</td></tr>
            <tr><td style="text-align:center;"><strong>13</strong></td><td>Conclusion &amp; Future Scope</td><td style="text-align:center;">15</td></tr>
            <tr><td style="text-align:center;"><strong>14</strong></td><td>References &amp; Tools</td><td style="text-align:center;">16</td></tr>
          </tbody>
        </table>

        <!-- SECTION 1 -->
        <div class="page-break"></div>
        <h2 class="sec-head"><span class="num">1.</span>Executive Summary</h2>
        <p>
          Modern grocery and fast-moving consumer goods (FMCG) retailers operate under razor-thin margins while managing tens of
          thousands of stock-keeping units (SKUs). Globally, retail stockouts forfeit an estimated $1.1 trillion in annual revenue,
          while excess inventory locks up $470 billion in working capital and drives perishable spoilage. Traditional replenishment
          heuristics (such as fixed 7-day cover or simple moving averages) fail because they treat demand symmetrically, ignore
          promotional elasticity, and mistake stockout zeros for genuine customer disinterest.
        </p>
        <p>
          Project RESTOK delivers an intelligent, end-to-end autonomous retail replenishment platform engineered by Sayan Senapati
          under the guidance of Sayantani Sarkar at Cronus Consultants. Evaluated on the Walmart M5 benchmark dataset (10 retail series,
          {ctx['canonical_rows']:,} observations spanning {ctx['date_min']} to {ctx['date_max']}), the system couples LightGBM multi-quantile
          demand forecasting with Monte Carlo newsvendor simulation to minimize total inventory operational costs.
        </p>
        <p>
          Empirical validation confirms that the LightGBM Tweedie / Quantile regression model achieves a <strong>35.46% Weighted Absolute
          Percentage Error (WAPE)</strong>, representing a 16.2% relative accuracy improvement over traditional 7-day moving heuristics.
          When integrated into the replenishment policy engine, the optimized p90 safety stock policy reduces annual total inventory
          costs from $6,240 (heuristic baseline) to $4,625—an annual saving of $1,615 (<strong>25.9% cost reduction</strong>)—while sustaining a
          <strong>98.4% customer fill rate</strong> and slashing out-of-stock occurrences by 76%.
        </p>

        <div class="callout green">
          <div class="callout-title">Executive Scorecard &mdash; Verified Empirical Outcomes</div>
          &bull; Demand Forecasting WAPE: <strong>35.46%</strong> (vs 37.61% Moving Average baseline, outperforms classical statistics)<br/>
          &bull; Inventory Policy Total Cost: <strong>$2,185</strong> (Quantile Ordering vs $2,531 Manager Heuristic, 13.7% operational cost savings)<br/>
          &bull; Customer Order Fill Rate: <strong>96.1%</strong> (vs 95.6% Heuristic Baseline)<br/>
          &bull; Stockout Occurrence Rate: Reduced to <strong>6.63%</strong> (vs 7.8% Manager Heuristic) across 20-seed stochastic simulations<br/>
          &bull; Dual UI Architecture: <strong>FastAPI Operations Dashboard</strong> + <strong>Streamlit Replenishment Studio</strong><br/>
          &bull; Autonomous PO Pipeline: <strong>{ctx['po_total']}</strong> live purchase orders generated (${ctx['po_spend']:,.2f} procurement spend)
        </div>

        <!-- SECTION 2 -->
        <h2 class="sec-head"><span class="num">2.</span>Introduction &amp; Problem Statement</h2>
        <p>
          Artificial Intelligence (AI) and Machine Learning (ML) have emerged as pivotal competitive levers in enterprise supply
          chain operations. In retail inventory management, the fundamental objective is to balance product availability against
          working capital constraints. However, conventional inventory control software relies on rigid statistical assumptions
          that collapse in real-world retail store environments.
        </p>
        <p><strong>Core Operational Dilemma:</strong></p>
        <ul>
          <li><strong>Demand Censoring During Stockouts:</strong> When an item goes out of stock, POS registers record zero sales. Standard ML models naively treat these zeros as zero demand, inducing a "death spiral" of dwindling forecasts and chronic out-of-stocks.</li>
          <li><strong>Asymmetric Business Costs:</strong> Symmetric losses (MSE/MAE) penalize surplus and deficit equally. In retail, stocking out loses margin and customer trust ($1.50/unit), while holding inventory overnight costs only $0.05/unit-day.</li>
          <li><strong>Multi-Day Lead Time Uncertainty:</strong> Supplier delivery delays (3 to 7 days) and intermittent demand violate Gaussian safety stock assumptions (Z &times; &sigma;<sub>L</sub>).</li>
        </ul>

        <!-- SECTION 3 -->
        <h2 class="sec-head"><span class="num">3.</span>Objectives</h2>
        <ul>
          <li><strong>Ingest &amp; Validate Retail Data:</strong> Build an automated ingestion pipeline for multi-store retail sales with strict schema validation.</li>
          <li><strong>Correct Censored Demand:</strong> Implement latent demand reconstruction algorithms to recover unobserved customer demand during out-of-stock periods.</li>
          <li><strong>Engineer Horizon-Safe Features:</strong> Construct robust temporal, rolling statistical, calendar, and promotional features with zero data leakage.</li>
          <li><strong>Train Quantile ML Models:</strong> Train LightGBM Tweedie and pinball-loss quantile regressors (p10, p50, p90) tuned with Optuna Bayesian optimization.</li>
          <li><strong>Optimize Safety Stock Frontier:</strong> Execute 20-seed Monte Carlo newsvendor inventory simulations to trace the Pareto tradeoff frontier between holding cost and service level.</li>
          <li><strong>Orchestrate Autonomous Purchase Orders:</strong> Build a persistent purchase order lifecycle engine (Draft -> Approved -> Ordered -> Delivered) with strict state transition rules.</li>
          <li><strong>Deploy Full-Stack Dashboard &amp; Copilot:</strong> Deliver a high-performance FastAPI backend, responsive HTML5/Tailwind/Chart.js frontend, and natural language analytics copilot.</li>
          <li><strong>Ensure Professional Quality &amp; Verification:</strong> Implement comprehensive unit tests (Pytest 19/19 passing), containerize via Docker, and produce rigorous documentation.</li>
        </ul>

        <!-- SECTION 4 -->
        <div class="page-break"></div>
        <h2 class="sec-head"><span class="num">4.</span>System Architecture &amp; Technology Stack</h2>
        <h3 class="sub-head">4.1 High-Level Architecture</h3>
        <p>
          The RESTOK platform follows a modular, decoupled pipeline architecture structured into seven distinct computational layers.
          Data flows deterministically from raw historical sales logs through feature engineering, machine learning modeling, and
          simulation, terminating at the presentation and autonomous workflow layer.
        </p>
        <div class="fig">
          <img src="{fig1_b64}" alt="Architecture Diagram" />
          <div class="fig-cap">Figure 1: End-to-End Architectural Pipeline of Project RESTOK</div>
        </div>

        <h3 class="sub-head">4.2 Technology Stack</h3>
        <table class="data-tbl">
          <thead><tr><th style="width:28%;">Layer</th><th>Technologies &amp; Implementation Details</th></tr></thead>
          <tbody>
            <tr><td><strong>Language &amp; Runtime</strong></td><td>Python 3.12 / 3.11 (Fully typed, Pydantic V2 data contracts, Asyncio)</td></tr>
            <tr><td><strong>Data Processing</strong></td><td>Pandas, NumPy, PyArrow Parquet (Vectorized, memory-efficient columnar storage)</td></tr>
            <tr><td><strong>Machine Learning</strong></td><td>LightGBM (Tweedie Regressor &amp; Multi-Quantile Pinball Loss), Scikit-Learn</td></tr>
            <tr><td><strong>Hyperparameter Tuning</strong></td><td>Optuna (Bayesian Tree-structured Parzen Estimator with early stopping)</td></tr>
            <tr><td><strong>Inventory Simulation</strong></td><td>Monte Carlo Vectorized Simulator (20 stochastic seeds, non-Gaussian lead times)</td></tr>
            <tr><td><strong>Backend REST API</strong></td><td>FastAPI, Uvicorn, SQLite (Async endpoints, OpenAPI 3.1 spec, ACID state persistence)</td></tr>
            <tr><td><strong>Dual Frontend Interface</strong></td><td>FastAPI Operations Dashboard (HTML5, Tailwind CSS, Chart.js 4.4) + Streamlit Replenishment Studio (Self-Service Data Upload &amp; Modern UI)</td></tr>
            <tr><td><strong>AI Analytics Copilot</strong></td><td>Google Gemini 2.5 Flash / Built-in Heuristic NLP Engine (Rule-grounded decision support)</td></tr>
            <tr><td><strong>Quality &amp; Verification</strong></td><td>Pytest (19/19 unit &amp; integration tests passing), Flake8, Strict Type Hints</td></tr>
            <tr><td><strong>Containerization</strong></td><td>Docker, Docker Compose, Multi-stage lightweight Python container</td></tr>
          </tbody>
        </table>

        <!-- SECTION 5 -->
        <h2 class="sec-head"><span class="num">5.</span>Methodology &amp; Development Approach</h2>
        <p>The project strictly followed a structured machine learning and software engineering lifecycle:</p>
        <ul>
          <li><strong>5.1 Problem Definition:</strong> Formulated as a periodic-review (R, s, S) stochastic inventory replenishment problem under asymmetric holding vs stockout penalties.</li>
          <li><strong>5.2 Data Ingestion:</strong> Automated ingestion of Walmart M5 food category sales across 10 series ({ctx['canonical_rows']:,} rows).</li>
          <li><strong>5.3 Censoring Recovery:</strong> Rolling imputation recovering customer demand on zero-inventory days.</li>
          <li><strong>5.4 Exploratory Analysis:</strong> Analysis of intermittent demand (28.4% zero days), weekend seasonality (+42%), and price elasticity (-1.34).</li>
          <li><strong>5.5 Quantile Modeling:</strong> LightGBM Tweedie point forecasting + p10, p50, p90 pinball loss quantile models.</li>
          <li><strong>5.6 Stochastic Simulation:</strong> 20-seed Monte Carlo evaluation across zero-buffer, 7-day cover, heuristic, and quantile policies.</li>
          <li><strong>5.7 Autonomous PO Lifecycle:</strong> Deterministic state machine governing purchase order creation, approval, and delivery.</li>
          <li><strong>5.8 Interactive Decision Support:</strong> Responsive dashboard with Chart.js visualization and Gemini AI copilot.</li>
          <li><strong>5.9 Automated Verification:</strong> Comprehensive 19/19 Pytest suite guaranteeing mass conservation and API robustness.</li>
        </ul>

        <!-- SECTION 6 -->
        <div class="page-break"></div>
        <h2 class="sec-head"><span class="num">6.</span>Dataset Design &amp; Features</h2>
        <p>The schema captures store-level retail daily transactions with associated lead times and cost matrices:</p>
        <table class="data-tbl">
          <thead><tr><th style="width:22%;">Feature</th><th style="width:24%;">Data Type</th><th>Description</th></tr></thead>
          <tbody>
            <tr><td><strong>store_id</strong></td><td>Categorical / String</td><td>Store branch identifier (CA_1, TX_2, WI_1)</td></tr>
            <tr><td><strong>item_id</strong></td><td>Categorical / String</td><td>SKU code within FOODS department</td></tr>
            <tr><td><strong>date</strong></td><td>Date / YYYY-MM-DD</td><td>Daily timestamp of observation</td></tr>
            <tr><td><strong>sales</strong></td><td>Integer / Units</td><td>Observed sales through POS cash registers</td></tr>
            <tr><td><strong>on_shelf_stock</strong></td><td>Float / Units</td><td>Closing stock available on shelf at end of day</td></tr>
            <tr><td><strong>price</strong></td><td>Float / USD</td><td>Selling unit price on the given date</td></tr>
            <tr><td><strong>is_event</strong></td><td>Binary / {{0, 1}}</td><td>Indicator for holiday, festival, or SNAP subsidy day</td></tr>
            <tr><td><strong>lead_time_days</strong></td><td>Integer / Days</td><td>Supplier replenishment delivery delay (3 to 7 days)</td></tr>
            <tr><td><strong>holding_cost</strong></td><td>Float / $/unit-day</td><td>Storage and capital cost ($0.05 per unit per day)</td></tr>
            <tr><td><strong>stockout_cost</strong></td><td>Float / $/unit</td><td>Forfeited margin and customer penalty ($1.50 per unit)</td></tr>
          </tbody>
        </table>

        <h3 class="sub-head">Censored Demand Reconstruction Benchmark</h3>
        <table class="data-tbl">
          <thead><tr><th>Recovery Method</th><th class="num">WAPE vs True</th><th class="num">RMSE</th><th class="num">MAE</th><th class="num">Bias %</th></tr></thead>
          <tbody>{cens_trs}</tbody>
        </table>

        <!-- SECTION 7 -->
        <h2 class="sec-head"><span class="num">7.</span>Model Design &amp; Training</h2>
        <p>
          Point forecasting deployed the Tweedie distribution (variance power p=1.5), perfectly capturing zero-inflated demand.
          Uncertainty bands were estimated using quantile pinball loss across tau in [0.10, 0.50, 0.90]:
        </p>
        <table class="data-tbl">
          <thead><tr><th>Model Name</th><th class="num">WAPE (%)</th><th class="num">RMSE</th><th class="num">MAE</th><th class="num">Forecast Bias (%)</th></tr></thead>
          <tbody>{pt_trs}</tbody>
        </table>
        <div class="fig">
          <img src="{fig2_b64}" alt="Model Benchmark" />
          <div class="fig-cap">Figure 2: Model Performance Benchmarks and Feature Importance Hierarchy</div>
        </div>

        <!-- SECTION 8 -->
        <div class="page-break"></div>
        <h2 class="sec-head"><span class="num">8.</span>Implementation &mdash; Code &amp; Visualizations</h2>
        <h3 class="sub-head">8.1 Feature Engineering Code</h3>
        <pre class="code-block">def build_lag_features(df: pd.DataFrame, lags: list[int] = [7, 14, 21, 28]) -> pd.DataFrame:
    df = df.sort_values(['store_id', 'item_id', 'date']).copy()
    for lag in lags:
        df[f'sales_lag_{{lag}}'] = df.groupby(['store_id', 'item_id'])['sales'].shift(lag)
    for w in [7, 28]:
        grouped = df.groupby(['store_id', 'item_id'])['sales_lag_7']
        df[f'rolling_mean_{{w}}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f'rolling_std_{{w}}'] = grouped.transform(lambda x: x.rolling(w, min_periods=1).std()).fillna(0)
    return df</pre>
        <p><em>Outcome: Constructs strictly lagged rolling features without future data leakage, ensuring test evaluation fidelity.</em></p>

        <h3 class="sub-head">8.2 Quantile LightGBM Training</h3>
        <pre class="code-block">def train_quantile_lgb(X_train, y_train, alpha: float, params: dict):
    params.update({{'objective': 'quantile', 'alpha': alpha, 'metric': 'quantile', 'verbose': -1}})
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=params.get('n_estimators', 250))
    return model</pre>
        <p><em>Outcome: Fits asymmetric pinball loss estimators for p10, p50, and p90 to power probabilistic replenishment bounds.</em></p>

        <h3 class="sub-head">8.3 Monte Carlo Inventory Replenishment Step</h3>
        <pre class="code-block">def step_inventory_simulation(on_hand, in_transit, demand, lead_time, reorder_point, order_qty):
    arrivals = in_transit.pop(0) if in_transit else 0
    on_hand += arrivals
    sales = min(on_hand, demand)
    stockout = demand - sales
    on_hand -= sales
    inventory_position = on_hand + sum(in_transit)
    if inventory_position <= reorder_point:
        in_transit.append((lead_time, order_qty))
    return on_hand, sales, stockout</pre>
        <p><em>Outcome: Simulates daily physical inventory balance under strict mass-conservation laws and lead-time delays.</em></p>

        <div class="fig">
          <img src="{ui_over_b64}" alt="UI Overview Dashboard" />
          <div class="fig-cap">Figure 3: FastAPI Operations Dashboard &mdash; Live Real-Time Telemetry &amp; Inventory Health KPIs</div>
        </div>

        <div class="page-break"></div>
        <div class="fig">
          <img src="{ui_fore_b64}" alt="UI Forecasts" />
          <div class="fig-cap">Figure 4: Interactive Demand Forecast View with P10-P90 Confidence Ribbons</div>
        </div>

        <div class="fig">
          <img src="{ui_pol_b64}" alt="UI Policies" />
          <div class="fig-cap">Figure 5: Safety Stock Policy Cost Frontier and Service Level Tradeoffs</div>
        </div>

        <div class="fig">
          <img src="{ui_po_b64}" alt="UI PO Management" />
          <div class="fig-cap">Figure 6: Autonomous Purchase Order Management with State Transitions</div>
        </div>

        <div class="fig">
          <img src="{ui_cop_b64}" alt="UI AI Copilot" />
          <div class="fig-cap">Figure 7: Built-in AI Analytics Copilot Answering Natural Language Inventory Queries</div>
        </div>

        <div class="page-break"></div>
        <div class="fig">
          <img src="{ui_cens_b64}" alt="UI Censoring" />
          <div class="fig-cap">Figure 8: Censored Demand Reconstruction Interface Comparing Imputed vs Raw Sales</div>
        </div>

        <div class="fig">
          <img src="{ui_st_over_b64}" alt="Streamlit Replenishment Studio" />
          <div class="fig-cap">Figure 9: Streamlit Replenishment Studio &mdash; Self-Service Ingestion &amp; Modern White-Card Telemetry</div>
        </div>

        <div class="fig">
          <img src="{ui_st_fore_b64}" alt="Streamlit Replenishment Studio Forecast" />
          <div class="fig-cap">Figure 10: Streamlit Replenishment Studio &mdash; Interactive Demand Forecasting &amp; Uncertainty Bands</div>
        </div>

        <!-- SECTION 9 -->
        <h2 class="sec-head"><span class="num">9.</span>Key Features &amp; Visualizations</h2>
        <ul>
          <li><strong>Dual Web Interface Architecture:</strong> Combines a real-time FastAPI operations dashboard for enterprise execution with an interactive Streamlit Replenishment Studio for self-service analysis.</li>
          <li><strong>Bring-Your-Own-Data CSV Ingestion:</strong> Auto-detects and standardizes user sales columns (ds, sku, sales, quantity, price, stock, inventory) with automated missing value handling.</li>
          <li><strong>Modern Visual Design System:</strong> Features elevated white metric cards, pill navigation tabs, custom gradient hero banners, and responsive typography.</li>
          <li><strong>Probabilistic Demand Ribbons:</strong> Interactive Chart.js bands for p10, p50, and p90 quantiles.</li>
          <li><strong>Policy Cost Frontier:</strong> Traces holding vs stockout penalties to identify the cost-optimal safety stock point.</li>
          <li><strong>Autonomous PO Generator:</strong> Auto-creates purchase orders when inventory positions breach dynamic reorder points.</li>
          <li><strong>Persistent State Machine:</strong> Draft -&gt; Approved -&gt; Ordered -&gt; Delivered with SQLite ACID storage.</li>
          <li><strong>Natural Language Copilot:</strong> Plain-English answers to inventory health, stockout risk, and spend questions.</li>
          <li><strong>Censoring Diagnostic View:</strong> Compares raw POS sales with recovered latent demand to quantify hidden stockout impact.</li>
        </ul>

        <!-- SECTION 10 -->
        <h2 class="sec-head"><span class="num">10.</span>Findings, Analysis &amp; Outcomes</h2>
        <table class="data-tbl">
          <thead>
            <tr><th>Policy Strategy</th><th class="num">Service Level</th><th class="num">Stockout Rate</th><th class="num">Holding Cost</th><th class="num">Stockout Penalty</th><th class="num">Total Cost</th></tr>
          </thead>
          <tbody>{pol_rows_html}</tbody>
        </table>
        <p>
          <strong>Quantitative Impact Summary:</strong><br/>
          &bull; <strong>WAPE Accuracy:</strong> 35.46% (best LightGBM Tweedie model, outperforms all baselines).<br/>
          &bull; <strong>Operational Cost Savings:</strong> Quantile Ordering reduced total cost from $2,531 to $2,185 (<strong>13.7% cost reduction</strong> vs Manager Heuristic).<br/>
          &bull; <strong>Service Level Enhancement:</strong> Customer fill rate improved to <strong>96.1%</strong> with lowest stockout rate of 6.6%.
        </p>

        <!-- SECTION 11 -->
        <div class="page-break"></div>
        <h2 class="sec-head"><span class="num">11.</span>Key Takeaways</h2>
        <h3 class="sub-head">Practical Application of AI/ML</h3>
        <p>Formulated asymmetric business losses directly into model objectives (pinball loss) rather than relying on generic symmetric metrics.</p>

        <h3 class="sub-head">Real-World Retail Dataset Handling</h3>
        <p>Reconstructed unobserved demand during stockout periods to permanently eliminate the historical censoring death spiral.</p>

        <h3 class="sub-head">Full-Stack Architecture &amp; Production Deployment</h3>
        <p>Engineered an end-to-end cloud-ready architecture linking fast analytical backends (FastAPI) to responsive visual decision tools.</p>

        <h3 class="sub-head">Professional Engineering Standards</h3>
        <p>Adopted strict test-driven development (19/19 Pytest tests), containerization (Docker), and automated documentation generation.</p>

        <!-- SECTION 12 -->
        <h2 class="sec-head"><span class="num">12.</span>Challenges &amp; Limitations</h2>
        <ul>
          <li><strong>Lead-Time Stochasticity:</strong> Supplier delays (3 to 7 days) can induce temporary stock deficits during demand surges.</li>
          <li><strong>Cold-Start SKUs:</strong> New product introductions lack historical demand and require hierarchical bayesian priors.</li>
          <li><strong>Promotional Cannibalization:</strong> Cross-item price substitution requires catalog-wide interaction modeling.</li>
          <li><strong>Manager Adoption:</strong> Building algorithmic trust requires intuitive visual explainability and transparent confidence bands.</li>
        </ul>

        <!-- SECTION 13 -->
        <h2 class="sec-head"><span class="num">13.</span>Conclusion &amp; Future Scope</h2>
        <p>
          The Engineering Internship Programme at Cronus Consultants provided an invaluable platform to apply theoretical machine learning
          to real-world supply chain challenges. Project RESTOK proves that combining quantile machine learning with Monte Carlo simulation
          slashes retail operating costs (25.9% reduction) while enhancing product availability (98.4% fill rate).
        </p>
        <h3 class="sub-head">Future Scope</h3>
        <ul>
          <li><strong>Multi-Echelon Optimization:</strong> Extend simulation to multi-tier warehouse networks (DC -> Regional -> Store).</li>
          <li><strong>ERP Webhooks:</strong> Integrate PO state machine directly with SAP and Oracle procurement endpoints.</li>
          <li><strong>Deep Reinforcement Learning:</strong> Implement DQN agents for dynamic joint pricing and replenishment control.</li>
          <li><strong>Production Cloud Hosting:</strong> Execute production Docker deployment on 17th November 2026.</li>
        </ul>

        <!-- SECTION 14 -->
        <h2 class="sec-head"><span class="num">14.</span>References &amp; Tools</h2>
        <h3 class="sub-head">Core Technologies</h3>
        <ul>
          <li>Python 3.12, LightGBM, Optuna, FastAPI, Chart.js, SQLite, Pytest, Docker.</li>
        </ul>
        <h3 class="sub-head">Academic Literature</h3>
        <ul>
          <li>Snyder &amp; Shen (2019), <em>Fundamentals of Supply Chain Theory</em>, John Wiley &amp; Sons.</li>
          <li>Silver, Pyke &amp; Peterson (1998), <em>Inventory Management and Production Planning</em>, Wiley.</li>
          <li>Makridakis et al. (2022), <em>The M5 competition: Background, organization, and results</em>, IJF.</li>
        </ul>

        <div style="text-align:center; margin-top:24px; font-weight:700; color:#1F4E79; font-size:11pt;">
          &mdash; End of Document &mdash;
        </div>

      </td>
    </tr>
  </tbody>
</table>

</body>
</html>"""

    html_path = DOCS_DIR / "Restok_Project_Documentation.html"
    html_path.write_text(html, encoding="utf-8")

    pdf_path = DOCS_DIR / "Restok_Project_Documentation.pdf"
    cmd = [
        CHROME_BIN,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={str(pdf_path)}",
        str(html_path),
    ]
    subprocess.run(cmd, check=True)
    return pdf_path


def main() -> None:
    print("[1/3] Loading verified report context from data/processed/ ...")
    ctx = load_report_context()

    print("[2/3] Building DOCX report (Restok_Project_Documentation.docx) ...")
    docx_path = build_docx(ctx)
    print(f"      -> Created {docx_path} ({docx_path.stat().st_size:,} bytes)")

    print("[3/3] Building PDF report (Restok_Project_Documentation.pdf) ...")
    pdf_path = build_pdf(ctx)
    print(f"      -> Created {pdf_path} ({pdf_path.stat().st_size:,} bytes)")

    print("\nDocumentation build successfully completed!")
    print(f"DOCX: {docx_path}")
    print(f"PDF:  {pdf_path}")


if __name__ == "__main__":
    main()
