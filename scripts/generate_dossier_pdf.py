"""Genere DOSSIER_PREPA_SOUTENANCE.pdf a partir de DOSSIER_PREPA_SOUTENANCE.md.

Dossier de revision des 3 projets (Blocs 1 & 2, RNCP40875) :
explications techniques + questions/reponses du jury.

Usage : python scripts/generate_dossier_pdf.py
Sortie : DOSSIER_PREPA_SOUTENANCE.pdf (a la racine du projet)
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
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

SOURCE_MD = ROOT / "DOSSIER_PREPA_SOUTENANCE.md"
OUTPUT = ROOT / "DOSSIER_PREPA_SOUTENANCE.pdf"

PRIMARY = colors.HexColor("#12395a")
ACCENT = colors.HexColor("#1f7a8c")
LIGHT = colors.HexColor("#eef3f7")
ANSWER_BG = colors.HexColor("#eaf4f7")
QUESTION_BG = colors.HexColor("#f4eee6")
DARK = colors.HexColor("#13293d")
GREY = colors.HexColor("#5b6b7a")
RULE = colors.HexColor("#cdd8e2")


# --------------------------------------------------------------------------- #
# Polices : on tente DejaVu (livre avec matplotlib) pour un support Unicode
# complet (fleches, tirets longs, guillemets...). Repli Helvetica sinon.
# --------------------------------------------------------------------------- #
def register_fonts():
    try:
        import matplotlib
        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        pdfmetrics.registerFont(TTFont("UDejaVu", str(ttf / "DejaVuSans.ttf")))
        pdfmetrics.registerFont(TTFont("UDejaVu-Bold", str(ttf / "DejaVuSans-Bold.ttf")))
        pdfmetrics.registerFont(TTFont("UDejaVu-Obl", str(ttf / "DejaVuSans-Oblique.ttf")))
        pdfmetrics.registerFont(TTFont("UDejaVu-Mono", str(ttf / "DejaVuSansMono.ttf")))
        pdfmetrics.registerFontFamily(
            "UDejaVu", normal="UDejaVu", bold="UDejaVu-Bold",
            italic="UDejaVu-Obl", boldItalic="UDejaVu-Bold",
        )
        return "UDejaVu", "UDejaVu-Bold", "UDejaVu-Obl", "UDejaVu-Mono", True
    except Exception as exc:  # pragma: no cover
        print("[warn] DejaVu indisponible, repli Helvetica :", exc)
        return "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Courier", False


BASE, BOLD, ITAL, MONO, UNICODE_OK = register_fonts()

# Caracteres a neutraliser si on n'a PAS de police Unicode.
_SANITIZE = {
    "→": "->", "←": "<-", "≈": "~", "≡": "=",
    "•": "-", "·": "-", "—": "-", "–": "-",
    "«": '"', "»": '"', "…": "...", " ": " ",
}


def uni(text: str) -> str:
    if UNICODE_OK:
        return text
    for bad, good in _SANITIZE.items():
        text = text.replace(bad, good)
    return text


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def build_styles():
    ss = getSampleStyleSheet()
    st = {}
    st["title"] = ParagraphStyle("title", parent=ss["Title"], fontName=BOLD, fontSize=25,
                                 textColor=PRIMARY, leading=30, alignment=TA_CENTER)
    st["subtitle"] = ParagraphStyle("subtitle", parent=ss["Normal"], fontName=BASE, fontSize=13,
                                    textColor=ACCENT, leading=18, alignment=TA_CENTER, spaceBefore=6)
    st["covermeta"] = ParagraphStyle("covermeta", parent=ss["Normal"], fontName=BASE, fontSize=10.5,
                                     textColor=GREY, leading=16, alignment=TA_CENTER)
    st["h1"] = ParagraphStyle("h1", parent=ss["Heading1"], fontName=BOLD, fontSize=15,
                              textColor=colors.white, leading=20, spaceBefore=16, spaceAfter=10,
                              backColor=PRIMARY, borderPadding=(7, 8, 7, 8))
    st["h2"] = ParagraphStyle("h2", parent=ss["Heading2"], fontName=BOLD, fontSize=12.3,
                              textColor=PRIMARY, leading=16, spaceBefore=11, spaceAfter=5)
    st["h3"] = ParagraphStyle("h3", parent=ss["Heading3"], fontName=BOLD, fontSize=10.8,
                              textColor=ACCENT, leading=14, spaceBefore=8, spaceAfter=3)
    st["h4"] = ParagraphStyle("h4", parent=ss["Heading4"], fontName=BOLD, fontSize=10,
                              textColor=DARK, leading=13, spaceBefore=6, spaceAfter=2)
    st["body"] = ParagraphStyle("body", parent=ss["Normal"], fontName=BASE, fontSize=9.7,
                                textColor=DARK, leading=14, alignment=TA_JUSTIFY, spaceAfter=6)
    st["quote"] = ParagraphStyle("quote", parent=st["body"], textColor=GREY, leftIndent=12,
                                 fontName=ITAL, spaceBefore=2, spaceAfter=7)
    st["bullet"] = ParagraphStyle("bullet", parent=st["body"], leftIndent=16, bulletIndent=4, spaceAfter=3)
    st["question"] = ParagraphStyle("question", parent=st["body"], fontName=BASE, fontSize=10,
                                    textColor=PRIMARY, alignment=TA_JUSTIFY, spaceBefore=10, spaceAfter=3,
                                    backColor=QUESTION_BG, borderPadding=(5, 6, 5, 6),
                                    borderColor=colors.HexColor("#d8c7ad"), borderWidth=0.5)
    st["answer"] = ParagraphStyle("answer", parent=st["body"], fontSize=9.6, leading=14,
                                  textColor=DARK, backColor=ANSWER_BG, borderPadding=(6, 7, 6, 7),
                                  borderColor=ACCENT, borderWidth=0.5, leftIndent=0, spaceAfter=9)
    st["code"] = ParagraphStyle("code", parent=ss["Normal"], fontName=MONO, fontSize=8.1,
                                textColor=DARK, backColor=LIGHT, leading=11, borderPadding=(6, 6, 6, 6),
                                spaceAfter=8)
    st["cell"] = ParagraphStyle("cell", parent=ss["Normal"], fontName=BASE, fontSize=8.2,
                                textColor=DARK, leading=11)
    st["cellh"] = ParagraphStyle("cellh", parent=st["cell"], fontName=BOLD, textColor=colors.white)
    return st


# --------------------------------------------------------------------------- #
# Inline markdown + echappement
# --------------------------------------------------------------------------- #
def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def inline_md(text: str) -> str:
    text = esc(uni(text))
    # 1) On protege les spans de code `...` AVANT bold/italic, sinon un '*'
    #    a l'interieur d'un code (ex. `gold_*`) casse le parseur italique.
    codes: list[str] = []

    def _stash(m):
        codes.append(m.group(1))
        return "\x00%d\x00" % (len(codes) - 1)

    text = re.sub(r"`([^`]+)`", _stash, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    def _restore(m):
        return '<font face="%s" backColor="#eef3f7">%s</font>' % (MONO, codes[int(m.group(1))])

    text = re.sub(r"\x00(\d+)\x00", _restore, text)
    return text


def make_table(header, rows, col_widths, S):
    data = [[Paragraph(inline_md(h), S["cellh"]) for h in header]]
    for r in rows:
        data.append([Paragraph(inline_md(c), S["cell"]) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("GRID", (0, 0), (-1, -1), 0.4, RULE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]))
    return t


def col_widths_for(header, avail=17.0 * cm):
    """Premiere colonne plus etroite si c'est un libelle court (Compétence, #, ...)."""
    n = len(header)
    short = {"compétence", "competence", "#", "outil / technique", "indicateur",
             "modèle", "modele", "critère", "critere", "slide(s)", "projet", "bloc"}
    first = header[0].strip().lower()
    if n >= 3 and first in short:
        w0 = avail * 0.16
        rest = (avail - w0) / (n - 1)
        return [w0] + [rest] * (n - 1)
    return [avail / n] * n


# --------------------------------------------------------------------------- #
# Parseur markdown
# --------------------------------------------------------------------------- #
def parse_markdown(lines, S):
    story = []
    i, n = 0, len(lines)
    title_done = False
    subtitle_pending = False

    def clear_pending():
        nonlocal subtitle_pending
        subtitle_pending = False

    while i < n:
        line = lines[i].rstrip("\n")
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped == "---":
            story.append(Spacer(1, 3))
            i += 1
            continue

        if stripped.startswith("```"):
            clear_pending()
            i += 1
            code_lines = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i].rstrip("\n"))
                i += 1
            i += 1
            code_text = esc(uni("\n".join(code_lines)))
            story.append(Paragraph(code_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), S["code"]))
            continue

        if stripped.startswith("#### "):
            clear_pending()
            story.append(Paragraph(inline_md(stripped[5:].strip()), S["h4"]))
            i += 1
            continue

        if stripped.startswith("### "):
            clear_pending()
            story.append(Paragraph(inline_md(stripped[4:].strip()), S["h3"]))
            i += 1
            continue

        if stripped.startswith("## "):
            text = stripped[3:].strip()
            if subtitle_pending:
                story.append(Paragraph(inline_md(text), S["subtitle"]))
                story.append(Spacer(1, 0.8 * cm))
                story.append(HRFlowable(width="60%", thickness=0.8, color=RULE,
                                        spaceBefore=4, spaceAfter=18, hAlign="CENTER"))
                story.append(Paragraph(
                    "Explications techniques &amp; Questions / Réponses du jury<br/>"
                    "Session 2026-2027", S["covermeta"]))
                story.append(PageBreak())
                subtitle_pending = False
            else:
                story.append(Paragraph(inline_md(text), S["h2"]))
            i += 1
            continue

        if stripped.startswith("# "):
            text = stripped[2:].strip()
            if not title_done:
                story.append(Spacer(1, 5.5 * cm))
                story.append(Paragraph(inline_md(text), S["title"]))
                title_done = True
                subtitle_pending = True
            else:
                clear_pending()
                story.append(Paragraph(inline_md(text), S["h1"]))
            i += 1
            continue

        if stripped.startswith(">"):
            clear_pending()
            quote_lines = []
            while i < n and lines[i].strip().startswith(">"):
                quote_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            story.append(Paragraph(inline_md(" ".join(quote_lines)), S["quote"]))
            continue

        if stripped.startswith("|"):
            clear_pending()
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = [
                [c.strip() for c in row.strip("|").split("|")]
                for row in table_lines
                if not re.match(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$", row)
            ]
            if rows:
                header, *body = rows
                story.append(make_table(header, body, col_widths_for(header), S))
                story.append(Spacer(1, 6))
            continue

        if re.match(r"^[-*]\s+", stripped):
            clear_pending()
            bullet_lines = []
            while i < n and re.match(r"^[-*]\s+", lines[i].strip()):
                bullet_lines.append(re.sub(r"^[-*]\s+", "", lines[i].strip()))
                i += 1
            for item in bullet_lines:
                story.append(Paragraph("&bull;&nbsp;&nbsp;" + inline_md(item), S["bullet"]))
            story.append(Spacer(1, 4))
            continue

        if re.match(r"^\d+\.\s+", stripped):
            clear_pending()
            num_lines = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].strip()):
                num_lines.append(lines[i].strip())
                i += 1
            for item in num_lines:
                story.append(Paragraph(inline_md(item), S["bullet"]))
            story.append(Spacer(1, 4))
            continue

        # Paragraphe : on regroupe les lignes contigues
        clear_pending()
        para_lines = [stripped]
        i += 1
        while i < n and lines[i].strip() and not re.match(
            r"^(#|>|\||[-*]\s|\d+\.\s|```|---|-> )", lines[i].strip()
        ):
            para_lines.append(lines[i].strip())
            i += 1
        joined = " ".join(para_lines)

        if joined.startswith("-> "):
            ans = inline_md(joined[3:].strip())
            label = '<b><font color="#1f7a8c">Réponse.</font></b> '
            story.append(Paragraph(label + ans, S["answer"]))
        elif joined.startswith("**Q"):
            story.append(Paragraph(inline_md(joined), S["question"]))
        else:
            story.append(Paragraph(inline_md(joined), S["body"]))

    return story


# --------------------------------------------------------------------------- #
# Mise en page
# --------------------------------------------------------------------------- #
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont(BASE, 7.6)
    canvas.setFillColor(GREY)
    if doc.page > 1:
        canvas.drawString(2 * cm, 1.1 * cm,
                          uni("Dossier de préparation — Soutenance Blocs 1 & 2 — RNCP40875"))
        canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, "Page %d" % doc.page)
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.restoreState()


def main():
    if not SOURCE_MD.exists():
        raise SystemExit(f"Fichier source introuvable : {SOURCE_MD}")

    S = build_styles()
    lines = SOURCE_MD.read_text(encoding="utf-8").splitlines()
    story = parse_markdown(lines, S)

    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.9 * cm, bottomMargin=1.8 * cm,
        title="Dossier de préparation - Soutenance Blocs 1 & 2 - RNCP40875",
        author="M1 Data Engineering & IA - EFREI",
    )
    frame = Frame(2 * cm, 1.8 * cm, A4[0] - 4 * cm, A4[1] - 3.7 * cm, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=on_page)])
    doc.build(story)
    print("PDF genere :", OUTPUT)


if __name__ == "__main__":
    main()
