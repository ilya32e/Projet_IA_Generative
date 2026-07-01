"""Convertit DOCUMENTATION_SOUTENANCE.md en PDF (meme charte graphique que le rapport).

Usage : python scripts/generate_doc_pdf.py
Sortie : DOCUMENTATION_SOUTENANCE.pdf (a la racine du projet)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

SOURCE_MD = ROOT / "DOCUMENTATION_SOUTENANCE.md"
OUTPUT = ROOT / "DOCUMENTATION_SOUTENANCE.pdf"

PRIMARY = colors.HexColor("#12395a")
ACCENT = colors.HexColor("#1f7a8c")
LIGHT = colors.HexColor("#eef3f7")
DARK = colors.HexColor("#13293d")
GREY = colors.HexColor("#5b6b7a")


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def build_styles():
    ss = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                                      fontSize=24, textColor=PRIMARY, leading=28, alignment=TA_CENTER)
    styles["subtitle"] = ParagraphStyle("subtitle", parent=ss["Normal"], fontSize=12, textColor=ACCENT,
                                         leading=17, alignment=TA_CENTER, spaceBefore=6)
    styles["h1"] = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=14.5,
                                   textColor=colors.white, leading=19, spaceBefore=14, spaceAfter=9,
                                   backColor=PRIMARY, borderPadding=(6, 8, 6, 8))
    styles["h2"] = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12,
                                   textColor=PRIMARY, leading=15, spaceBefore=10, spaceAfter=5)
    styles["h3"] = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=10.6,
                                   textColor=ACCENT, leading=14, spaceBefore=7, spaceAfter=3)
    styles["body"] = ParagraphStyle("body", parent=ss["Normal"], fontSize=9.8, textColor=DARK,
                                     leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
    styles["quote"] = ParagraphStyle("quote", parent=styles["body"], textColor=GREY, leftIndent=10,
                                      borderColor=ACCENT, borderWidth=0, fontName="Helvetica-Oblique")
    styles["bullet"] = ParagraphStyle("bullet", parent=styles["body"], leftIndent=14, bulletIndent=4, spaceAfter=3)
    styles["code"] = ParagraphStyle("code", parent=ss["Normal"], fontName="Courier", fontSize=8.4,
                                     textColor=DARK, backColor=LIGHT, leading=11.5, borderPadding=(6, 6, 6, 6),
                                     spaceAfter=8)
    styles["cell"] = ParagraphStyle("cell", parent=ss["Normal"], fontSize=8.4, textColor=DARK, leading=11.5)
    styles["cellh"] = ParagraphStyle("cellh", parent=styles["cell"], fontName="Helvetica-Bold", textColor=colors.white)
    return styles


S = build_styles()


# --------------------------------------------------------------------------- #
# Helpers d'echappement et d'inline markdown
# --------------------------------------------------------------------------- #
def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline_md(text: str) -> str:
    text = esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" backColor="#eef3f7">\1</font>', text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def make_table(header, rows, col_widths, header_bg=PRIMARY):
    data = [[Paragraph(inline_md(h), S["cellh"]) for h in header]]
    for r in rows:
        data.append([Paragraph(inline_md(c), S["cell"]) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, PRIMARY),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cdd8e2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]
    t.setStyle(TableStyle(style))
    return t


# --------------------------------------------------------------------------- #
# Parseur Markdown minimal (sous-ensemble utilise par DOCUMENTATION_SOUTENANCE.md)
# --------------------------------------------------------------------------- #
def parse_markdown(lines: list[str]) -> list:
    story = []
    i = 0
    n = len(lines)
    title_done = False

    while i < n:
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 4))
            i += 1
            continue

        if stripped.startswith("```"):
            i += 1
            code_lines = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            i += 1
            code_text = "\n".join(code_lines)
            story.append(Paragraph(esc(code_text).replace("\n", "<br/>").replace(" ", "&nbsp;"), S["code"]))
            continue

        if stripped.startswith("# "):
            text = stripped[2:].strip()
            if not title_done:
                story.append(Spacer(1, 4 * cm))
                story.append(Paragraph(esc(text), S["title"]))
                title_done = True
            else:
                story.append(Paragraph(esc(text), S["h1"]))
            i += 1
            continue

        if stripped.startswith("## "):
            story.append(Paragraph(inline_md(stripped[3:].strip()), S["h1"]))
            i += 1
            continue

        if stripped.startswith("### "):
            story.append(Paragraph(inline_md(stripped[4:].strip()), S["h2"]))
            i += 1
            continue

        if stripped.startswith("> "):
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            story.append(Paragraph(inline_md(" ".join(quote_lines)), S["quote"]))
            continue

        if stripped.startswith("|"):
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = [
                [cell.strip() for cell in row.strip("|").split("|")]
                for row in table_lines
                if not re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$", row)
            ]
            if rows:
                header, *body = rows
                col_count = len(header)
                avail_width = 17.0 * cm
                col_widths = [avail_width / col_count] * col_count
                story.append(make_table(header, body, col_widths))
                story.append(Spacer(1, 6))
            continue

        if re.match(r"^[-*]\s+", stripped):
            bullet_lines = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                bullet_lines.append(re.sub(r"^[-*]\s+", "", lines[i].strip()))
                i += 1
            for item in bullet_lines:
                story.append(Paragraph("&bull;&nbsp;&nbsp;" + inline_md(item), S["bullet"]))
            story.append(Spacer(1, 4))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            number_lines = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                number_lines.append(lines[i].strip())
                i += 1
            for item in number_lines:
                story.append(Paragraph(inline_md(item), S["bullet"]))
            story.append(Spacer(1, 4))
            continue

        # Paragraphe normal : on regroupe les lignes contigues
        para_lines = [stripped]
        i += 1
        while i < n and lines[i].strip() and not re.match(
            r"^(#|>|\||[-*]\s|\d+\.\s|```|---)", lines[i].strip()
        ):
            para_lines.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline_md(" ".join(para_lines)), S["body"]))

    return story


# --------------------------------------------------------------------------- #
# Mise en page
# --------------------------------------------------------------------------- #
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.1 * cm, "AISCA - Documentation technique (soutenance)")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, "Page %d" % doc.page)
    canvas.setStrokeColor(colors.HexColor("#cdd8e2"))
    canvas.setLineWidth(0.4)
    canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.restoreState()


def main():
    if not SOURCE_MD.exists():
        raise SystemExit(f"Fichier source introuvable : {SOURCE_MD}")

    lines = SOURCE_MD.read_text(encoding="utf-8").splitlines()
    story = parse_markdown(lines)

    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=1.8 * cm,
        title="AISCA - Documentation technique", author="Salah-Eddine Zbiri",
    )
    frame_main = Frame(2 * cm, 1.8 * cm, A4[0] - 4 * cm, A4[1] - 3.8 * cm, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame_main], onPage=on_page)])

    # Saut de page apres le titre principal pour demarrer le sommaire/contenu propre
    final_story = []
    title_seen = False
    for item in story:
        final_story.append(item)
        if not title_seen and isinstance(item, Paragraph) and item.style.name == "title":
            title_seen = True
            final_story.append(Spacer(1, 1 * cm))
            final_story.append(PageBreak())

    doc.build(final_story)
    print("PDF genere :", OUTPUT)


if __name__ == "__main__":
    main()
