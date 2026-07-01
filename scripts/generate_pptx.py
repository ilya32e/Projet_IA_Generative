# -*- coding: utf-8 -*-
"""
Génère la présentation PowerPoint de soutenance (3 projets), version conforme
au Guide de préparation RNCP40875 (Soutenance Bloc 1 & 2, Efrei).

Conformité guide :
  - ordre imposé : Projet 1 = Architecture de données (Bloc 1),
    Projet 2 = Data Science (Bloc 2), Projet 3 = IA générative (Bloc 2) ;
  - 20 slides exactement pour les 30 minutes chronométrées (5 slides/9 min pour
    P1, 5/9 min pour P2, 3/6 min pour P3, conforme au tableau du guide) ;
  - slide « Répartition des contributions individuelles » ;
  - mapping explicite aux compétences C1.1 → C5.3, avec une ligne « → Cx.y car... »
    visible sur chaque slide de contenu (logique besoin → choix → preuve →
    résultat → limites).

Les preuves détaillées qui ne rentrent pas dans ces 20 slides (charts de
performance, tableaux comparatifs, démonstration GenAI intégrale) sont dans un
fichier séparé : voir scripts/generate_annexes_pptx.py →
Annexes_Preuves_Techniques.pptx, à garder ouvert pendant les 20 minutes
d'échange avec le jury.

Sortie : Soutenance_3_Projets.pptx (16:9, sobre, académique, avec notes, 20 slides).
Usage  : python scripts/generate_pptx.py
Pré-requis images : lancer d'abord  python scripts/prep_images.py
"""

import os
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# --------------------------------------------------------------------------
# Charte graphique
# --------------------------------------------------------------------------
NAVY = RGBColor(0x14, 0x2A, 0x47)      # bleu nuit (fonds, titres)
NAVY2 = RGBColor(0x1F, 0x3A, 0x5F)
TEAL = RGBColor(0x2A, 0x9D, 0x8F)      # accent principal
TEAL_D = RGBColor(0x1E, 0x73, 0x6A)
GOLD = RGBColor(0xE9, 0xC4, 0x6A)
CORAL = RGBColor(0xE7, 0x6F, 0x51)
SLATE = RGBColor(0x2B, 0x33, 0x40)     # texte principal
GREY = RGBColor(0x5C, 0x66, 0x73)      # texte secondaire
FAINT = RGBColor(0x8A, 0x93, 0x9F)
LIGHT = RGBColor(0xF4, 0xF6, 0xF9)     # fond clair
CARD = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xD6, 0xDC, 0xE4)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

FONT = "Calibri"
FONT_H = "Calibri"

SW, SH = Inches(13.333), Inches(7.5)
IMG_DIR = (Path(__file__).resolve().parents[1] / "assets" / "screenshots")

prs = Presentation()
prs.slide_width = SW
prs.slide_height = SH
BLANK = prs.slide_layouts[6]

# accent par projet, dans l'ordre imposé par le guide
# P1 = Architecture (Bloc 1) · P2 = Data Science (Bloc 2) · P3 = IA générative (Bloc 2)
ACCENT = {"P1": TEAL, "P2": CORAL, "P3": GOLD}

_page = {"n": 0}


# --------------------------------------------------------------------------
# Helpers bas niveau
# --------------------------------------------------------------------------
def add_slide():
    return prs.slides.add_slide(BLANK)


def add_rect(slide, x, y, w, h, color, shape=MSO_SHAPE.RECTANGLE, line=None, lw=1.0):
    shp = slide.shapes.add_shape(shape, int(x), int(y), int(w), int(h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(lw)
    shp.shadow.inherit = False
    return shp


def textbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(int(x), int(y), int(w), int(h))
    tb.text_frame.word_wrap = True
    return tb, tb.text_frame


def R(p, text, size, color=SLATE, bold=False, italic=False, font=FONT):
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font
    return run


def footer(slide, tag, accent):
    _page["n"] += 1
    # filet + numéro de page + tag projet en pied
    add_rect(slide, Inches(0.55), Inches(7.07), Inches(12.23), Pt(1.2), LINE)
    tb, tf = textbox(slide, Inches(0.55), Inches(7.12), Inches(8), Inches(0.3))
    R(tf.paragraphs[0], tag, 9, FAINT)
    tb, tf = textbox(slide, Inches(11.6), Inches(7.12), Inches(1.2), Inches(0.3))
    pp = tf.paragraphs[0]; pp.alignment = PP_ALIGN.RIGHT
    R(pp, f"{_page['n']:02d}", 9, accent, bold=True)


def header(slide, kicker, title, accent=TEAL, foot="Soutenance · RNCP40875 · Blocs 1 & 2"):
    add_rect(slide, 0, 0, SW, Inches(1.18), NAVY)
    add_rect(slide, 0, Inches(1.18), SW, Pt(3.5), accent)
    add_rect(slide, Inches(0.55), Inches(0.34), Pt(5), Inches(0.52), accent)
    tb, tf = textbox(slide, Inches(0.75), Inches(0.14), Inches(11.8), Inches(1.0))
    R(tf.paragraphs[0], kicker.upper(), 11.5, GOLD, bold=True)
    p2 = tf.add_paragraph()
    R(p2, title, 25, WHITE, bold=True, font=FONT_H)
    footer(slide, foot, accent)


def bullets(tf, items, size=16):
    """items: (level, text, bold)."""
    first = True
    for level, text, bold in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(5)
        p.space_before = Pt(1)
        if level == 0:
            R(p, "▸ ", size, ACCENT_CUR[0], bold=True)
            R(p, text, size, NAVY2 if bold else SLATE, bold=bold)
        else:
            R(p, "•  ", size - 2, FAINT)
            R(p, text, size - 2, GREY)


ACCENT_CUR = [TEAL]  # accent courant (mutable)


def bullets_box(slide, items, x, y, w, h, size=16):
    tb, tf = textbox(slide, x, y, w, h)
    bullets(tf, items, size=size)
    return tb


def pic(slide, name, bx, by, bw, bh, frame=True):
    """Insère une image (nom de fichier, version _t.png), ajustée dans la boîte."""
    path = IMG_DIR / name
    if not path.exists():
        # fallback : placeholder
        ph = add_rect(slide, bx, by, bw, bh, LIGHT, line=LINE)
        tb, tf = textbox(slide, bx, by + bh / 2 - Inches(0.2), bw, Inches(0.4))
        pp = tf.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER
        R(pp, f"[{name}]", 11, FAINT, italic=True)
        return
    iw, ih = Image.open(path).size
    scale = min(bw / iw, bh / ih)
    w = int(iw * scale); h = int(ih * scale)
    x = int(bx + (bw - w) / 2); y = int(by + (bh - h) / 2)
    off = Emu(45720)  # 0.05"
    if frame:
        add_rect(slide, x + off, y + off, w, h, RGBColor(0xC9, 0xD0, 0xDA))  # ombre
        pad = Emu(28000)
        add_rect(slide, x - pad, y - pad, w + 2 * pad, h + 2 * pad, CARD, line=LINE, lw=1.0)
    p = slide.shapes.add_picture(str(path), x, y, w, h)
    p.line.color.rgb = RGBColor(0xE4, 0xE8, 0xEE)
    p.line.width = Pt(0.5)
    p.shadow.inherit = False
    return p


def img_caption(slide, text, x, y, w):
    tb, tf = textbox(slide, x, y, w, Inches(0.32))
    pp = tf.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER
    R(pp, text, 10, FAINT, italic=True)


def split(slide, items, image=None, cap=None, size=16, text_w=Inches(5.55)):
    """Layout texte (gauche) + image (droite)."""
    bullets_box(slide, items, Inches(0.7), Inches(1.55), text_w, Inches(5.2), size=size)
    if image:
        ix = Inches(0.7) + text_w + Inches(0.35)
        iw = SW - ix - Inches(0.55)
        pic(slide, image, ix, Inches(1.62), iw, Inches(4.75))
        if cap:
            img_caption(slide, cap, ix, Inches(6.5), iw)


def table(slide, rows, top, left, width, height, col_w=None, fs=13, accent=TEAL):
    nr, nc = len(rows), len(rows[0])
    t = slide.shapes.add_table(nr, nc, int(left), int(top), int(width), int(height)).table
    if col_w:
        for i, cw in enumerate(col_w):
            t.columns[i].width = int(cw)
    for r in range(nr):
        for c in range(nc):
            cell = t.cell(r, c)
            cell.margin_left = Inches(0.09); cell.margin_right = Inches(0.07)
            cell.margin_top = Inches(0.03); cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.text_frame.word_wrap = True
            head = (r == 0)
            lines = rows[r][c].split("\n")
            for li, line in enumerate(lines):
                para = cell.text_frame.paragraphs[0] if li == 0 else cell.text_frame.add_paragraph()
                R(para, line, fs + (1 if head else 0),
                  WHITE if head else SLATE, bold=head or c == 0)
            cell.fill.solid()
            if head:
                cell.fill.fore_color.rgb = NAVY
            else:
                cell.fill.fore_color.rgb = CARD if r % 2 else LIGHT
    return t


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def set_accent(a):
    ACCENT_CUR[0] = a


def comp_card(slide, bloc, comps, x, y, w, h, accent):
    """Carte « compétences démontrées » (codes du référentiel) pour les diviseurs."""
    add_rect(slide, x, y, w, h, RGBColor(0x1B, 0x32, 0x52), line=None)
    add_rect(slide, x, y, w, Inches(0.5), accent)
    tb, tf = textbox(slide, x + Inches(0.18), y + Inches(0.06), w - Inches(0.3), Inches(0.4))
    R(tf.paragraphs[0], bloc, 13, NAVY if accent is GOLD else WHITE, bold=True)
    tb, tf = textbox(slide, x + Inches(0.22), y + Inches(0.62), w - Inches(0.4), h - Inches(0.7))
    first = True
    for code, label in comps:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(4)
        R(p, f"{code}  ", 12.5, GOLD, bold=True)
        R(p, label, 12.5, RGBColor(0xDD, 0xE4, 0xEE))


def divider(num, title, subtitle, accent, bloc, comps):
    s = add_slide()
    add_rect(s, 0, 0, SW, SH, NAVY)
    add_rect(s, 0, 0, Inches(0.18), SH, accent)
    # gros numéro
    tb, tf = textbox(s, Inches(0.7), Inches(0.95), Inches(4), Inches(2.2))
    R(tf.paragraphs[0], num, 120, RGBColor(0x24, 0x3A, 0x57), bold=True, font=FONT_H)
    add_rect(s, Inches(0.85), Inches(3.15), Inches(2.2), Pt(4), accent)
    tb, tf = textbox(s, Inches(0.85), Inches(3.3), Inches(6.6), Inches(0.6))
    R(tf.paragraphs[0], f"PROJET {num}", 16, accent, bold=True)
    tb, tf = textbox(s, Inches(0.85), Inches(3.78), Inches(6.6), Inches(1.4))
    R(tf.paragraphs[0], title, 38, WHITE, bold=True, font=FONT_H)
    tb, tf = textbox(s, Inches(0.85), Inches(5.35), Inches(6.6), Inches(1.4))
    R(tf.paragraphs[0], subtitle, 15, RGBColor(0xC7, 0xD2, 0xDE), italic=True)
    # carte compétences à droite
    comp_card(s, bloc, comps, Inches(7.85), Inches(1.5), Inches(4.95), Inches(4.7), accent)
    return s


# ==========================================================================
# SLIDE 1 — TITRE, NOMS, BLOCS ÉVALUÉS
# ==========================================================================
s = add_slide()
add_rect(s, 0, 0, SW, SH, NAVY)
add_rect(s, 0, 0, SW, Inches(0.18), TEAL)
add_rect(s, 0, Inches(7.32), SW, Inches(0.18), GOLD)
add_rect(s, Inches(0.0), Inches(2.55), Inches(3.0), Pt(3), TEAL)

tb, tf = textbox(s, Inches(0.9), Inches(0.7), Inches(11.5), Inches(0.9))
R(tf.paragraphs[0], "SOUTENANCE — TITRE RNCP40875", 13, GOLD, bold=True)
p = tf.add_paragraph()
R(p, "Expert en Ingénierie des données · Blocs 1 & 2 · Efrei", 12.5, RGBColor(0xC7, 0xD2, 0xDE))

tb, tf = textbox(s, Inches(0.9), Inches(2.45), Inches(11.6), Inches(1.7))
R(tf.paragraphs[0], "De la donnée brute à la décision", 42, WHITE, bold=True, font=FONT_H)
p = tf.add_paragraph()
R(p, "Trois projets formatifs, deux blocs de compétences", 22, GOLD)

tb, tf = textbox(s, Inches(0.9), Inches(4.2), Inches(11.6), Inches(0.7))
R(tf.paragraphs[0],
  "Architecture de données (Bloc 1)   ·   Data Science (Bloc 2)   ·   IA générative (Bloc 2)",
  15, RGBColor(0xC7, 0xD2, 0xDE), italic=True)

tb, tf = textbox(s, Inches(0.9), Inches(5.35), Inches(11.6), Inches(1.7))
for label, val in [("Présenté par", "[Nom Prénom] · [Nom Prénom] · [Nom Prénom]"),
                   ("Formation", "Mastère 1 Data Engineering & IA — Efrei"),
                   ("Jury / Encadrant", "[Encadrant]"),
                   ("Date", "[Date]")]:
    p = tf.add_paragraph()
    R(p, f"{label} : ", 14, GOLD, bold=True)
    R(p, val, 14, WHITE)
notes(s, "Bonjour à tous. Nous allons vous présenter trois projets qui racontent en réalité la même "
         "histoire : partir d'une donnée brute, et arriver à une décision utile. Le titre que nous "
         "visons, Expert en Ingénierie des données, demande de maîtriser deux blocs de compétences. Le "
         "Bloc 1, c'est construire et superviser une architecture de données fiable. Le Bloc 2, c'est en "
         "tirer de la valeur : par la data science, et par l'intelligence artificielle générative. Pour "
         "le démontrer, nous avons construit trois projets concrets : une architecture de données sur le "
         "logement parisien, un système de prédiction marketing, et un assistant intelligent "
         "d'orientation professionnelle. La présentation est collective, mais chacun de nous pourra "
         "répondre individuellement sur l'ensemble du travail. Commençons par planter le décor : le "
         "contexte métier qui relie ces trois projets.")

# ==========================================================================
# SLIDE 2 — CONTEXTE MÉTIER ET PROBLÉMATIQUE
# ==========================================================================
set_accent(TEAL)
s = add_slide()
header(s, "Introduction", "Contexte métier & problématique globale")
left = [
    (0, "Cas d'usage global — aider la décision à partir de la donnée", True),
    (1, "RH : orienter un candidat vers les bons métiers", False),
    (1, "Marketing : arbitrer un budget publicitaire", False),
    (1, "Logement : comparer les territoires parisiens", False),
    (0, "Problématique commune", True),
    (1, "la valeur naît de la chaîne complète, pas du modèle seul", False),
    (1, "ingérer, transformer, modéliser, restituer — de façon fiable", False),
    (0, "Blocs évalués", True),
    (1, "Bloc 1 : construire & superviser une architecture de données", False),
    (1, "Bloc 2 : data science, modélisation et IA générative", False),
]
bullets_box(s, left, Inches(0.7), Inches(1.55), Inches(7.0), Inches(5.2), size=15.5)
# frise horizontale donnée -> décision
steps = ["Donnée", "Traitement", "Modèle", "Restitution", "Décision"]
add_rect(s, Inches(8.0), Inches(1.55), Inches(4.78), Inches(0.5), LIGHT, line=LINE)
tb, tf = textbox(s, Inches(8.15), Inches(1.6), Inches(4.5), Inches(0.4))
R(tf.paragraphs[0], "LE FIL ROUGE : donnée → décision", 12, TEAL, bold=True)
y = Inches(2.25)
for i, st in enumerate(steps):
    c = [TEAL, TEAL_D, NAVY2, CORAL, GOLD][i]
    box = add_rect(s, Inches(8.0), y, Inches(4.78), Inches(0.66), c, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    pp = box.text_frame.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER
    R(pp, st, 14, WHITE if i != 4 else NAVY, bold=True)
    y = Emu(int(y) + int(Inches(0.78)))
notes(s, "Trois métiers différents — RH, marketing, logement — mais une seule question de fond : comment "
         "aider quelqu'un à décider, à partir d'une donnée brute et souvent désordonnée ? Dans les trois "
         "cas, la valeur ne vient jamais du modèle tout seul. Elle vient de toute la chaîne : on ingère "
         "la donnée, on la transforme, on construit un modèle ou un moteur, on la restitue de façon "
         "lisible, et c'est seulement à ce moment-là que la décision devient possible. C'est ce fil "
         "rouge, donnée, traitement, modèle, restitution, décision, que vous allez retrouver dans les "
         "trois projets. Le Bloc 1 couvre surtout les deux premières étapes : construire et superviser "
         "l'architecture qui rend la donnée exploitable. Le Bloc 2 couvre les trois suivantes : "
         "modéliser, restituer, et permettre la décision, avec la data science et l'IA générative. "
         "Voyons maintenant comment nos trois projets se répartissent entre ces deux blocs.")

# ==========================================================================
# SLIDE 3 — VUE D'ENSEMBLE DES TROIS PROJETS
# ==========================================================================
s = add_slide()
header(s, "Vue d'ensemble", "Trois projets, deux blocs de compétences", TEAL)
table(s, [
    ["", "Projet 1 — Architecture", "Projet 2 — Data Science", "Projet 3 — IA générative"],
    ["Bloc", "Bloc 1", "Bloc 2", "Bloc 2"],
    ["Application", "Urban Data Explorer", "ROI Marketing", "AISCA (RH Tech)"],
    ["Cœur technique", "Bronze/Silver/Gold + géocodage", "4 modèles + SHAP", "SBERT + RAG contraint"],
    ["Stockage", "MySQL + MongoDB", "Artefacts modèle", "Référentiel + cache"],
    ["Restitution", "API + carte MapLibre", "API + dashboard", "App Streamlit"],
    ["Compétences", "C1.1–C2.4", "C3.1–C4.3", "C5.1–C5.3"],
], Inches(1.55), Inches(0.55), Inches(12.2), Inches(4.6),
   col_w=[Inches(1.9), Inches(3.5), Inches(3.4), Inches(3.4)], fs=12.5)
notes(s, "Voici la carte d'ensemble. Le projet 1, Urban Data Explorer, valide le Bloc 1 : il construit "
         "l'architecture, avec un data lake en trois zones et un système de géocodage, stocké dans deux "
         "bases complémentaires, MySQL et MongoDB, et restitué par une API et une carte interactive. Les "
         "projets 2 et 3 valident le Bloc 2, chacun à sa manière. Le projet 2, ROI Marketing, mobilise "
         "quatre modèles de machine learning et la méthode SHAP pour les expliquer, restitués dans un "
         "tableau de bord. Le projet 3, AISCA, mobilise un moteur sémantique SBERT et un agent génératif, "
         "restitués dans une application Streamlit. La dernière ligne du tableau résume tout : les "
         "compétences couvertes vont de C1.1 à C5.3, sans trou. Avant d'entrer dans le détail technique, "
         "un mot sur qui a fait quoi dans l'équipe.")

# ==========================================================================
# SLIDE 4 — RÉPARTITION DES CONTRIBUTIONS INDIVIDUELLES
# ==========================================================================
s = add_slide()
header(s, "Équipe", "Répartition des contributions individuelles", TEAL)
table(s, [
    ["Membre", "Projet 1 — Architecture", "Projet 2 — Data Science", "Projet 3 — IA générative"],
    ["[Étudiant 1]", "Pilote · Data Lake + bases", "Préparation & EDA", "Moteur sémantique"],
    ["[Étudiant 2]", "API + streaming", "Modélisation & sélection", "Agent RAG"],
    ["[Étudiant 3]", "Géocodage + restitution", "Explicabilité & dashboard", "Évaluation & calibrage"],
], Inches(1.7), Inches(0.7), Inches(12.0), Inches(2.6),
   col_w=[Inches(2.2), Inches(3.4), Inches(3.2), Inches(3.2)], fs=12.5)
bullets_box(s, [
    (0, "Présentation collective, évaluation individualisée", True),
    (1, "chaque membre maîtrise l'ensemble du projet, pas seulement sa partie", False),
    (1, "chacun sait justifier un choix technique et citer ses compétences", False),
    (0, "À personnaliser avant la soutenance (noms & rôles réels)", True),
], Inches(0.7), Inches(4.0), Inches(12), Inches(2.6), size=15)
notes(s, "La présentation est collective, mais l'évaluation reste individuelle, donc ce tableau compte "
         "vraiment. Il répartit nos contributions, projet par projet et personne par personne — à "
         "adapter avec vos noms et vos rôles réels avant la vraie soutenance. Le point important n'est "
         "pas seulement qui a fait quoi : c'est que chacun de nous maîtrise l'ensemble du projet, sait "
         "justifier au moins un choix technique personnel, sait citer les compétences qu'il démontre, et "
         "peut discuter une limite si le jury le demande. C'est exactement ce que vérifient les vingt "
         "minutes d'échange à la fin. Nous allons maintenant entrer dans le premier projet, dédié au "
         "Bloc 1 : l'architecture de données.")

# ==========================================================================
# PROJET 1 — ARCHITECTURE DE DONNÉES (BLOC 1) — slides 5 à 9
# ==========================================================================
set_accent(TEAL)
s = divider(
    "1", "Urban Data Explorer",
    "Architecture de données du logement parisien — ingestion, fusion, API & carto — Bloc 1",
    TEAL, "BLOC 1 — Compétences démontrées",
    [("C1.1", "Base relationnelle (MySQL)"),
     ("C1.2", "Base NoSQL (MongoDB / GeoJSON)"),
     ("C1.3", "Data Lake médaillon multi-sources"),
     ("C1.4", "Scalabilité & résilience"),
     ("C2.1", "API interopérable & sécurisée"),
     ("C2.2", "Streaming distribué (Kafka/Redis)"),
     ("C2.3", "Intégration multi-sources"),
     ("C2.4", "Optimisation & mesure des pipelines")])
notes(s, "Premier projet : Urban Data Explorer. Le Bloc 1, c'est savoir construire, surveiller et "
         "améliorer une architecture de données. On a choisi un exemple concret : rassembler une "
         "dizaine de sources de données ouvertes sur le logement à Paris, pour pouvoir enfin les "
         "comparer entre elles. Ce projet, à lui seul, touche les huit compétences du Bloc 1, de C1.1 à "
         "C2.4 : une base relationnelle, une base non relationnelle, un data lake, la scalabilité, une "
         "API sécurisée, le streaming, l'intégration de plusieurs sources, et la mesure de performance. "
         "On va montrer, une par une, comment on les a vraiment utilisées, en suivant un plan simple : "
         "d'abord le besoin, puis l'architecture, puis les choix techniques, puis les preuves, et enfin "
         "les résultats et les limites.")

# --- Slide 6 : 1+2 · besoin métier & architecture proposée ---
s = add_slide(); header(s, "Projet 1 · Architecture", "1+2 · Besoin métier & architecture proposée", TEAL)
bullets_box(s, [
    (0, "Besoin métier & gouvernance (C1.3)", True),
    (1, "~10 sources open data dispersées (DVF, INSEE, loyers, bruit/air, BAN…) ; aucune ne répond "
        "seule à « où vivre/investir, à quel prix réel ? »", False),
    (1, "gouvernance exigée : traçabilité jusqu'à la source, reproductibilité en une commande, "
        "qualité mesurée (pas affirmée)", False),
], Inches(0.7), Inches(1.5), Inches(11.9), Inches(1.3), size=14)
flow = [
    ("10 sources\nopen data", FAINT, NAVY),
    ("BRONZE\nbrut", RGBColor(0xB0, 0x7A, 0x43), WHITE),
    ("SILVER\nnettoyé", RGBColor(0x8A, 0x93, 0x9F), WHITE),
    ("GOLD\nparquet\n(year=)", GOLD, NAVY),
    ("MySQL +\nMongoDB", NAVY2, WHITE),
    ("API\nFastAPI", TEAL_D, WHITE),
    ("Carte +\nDashboard", TEAL, WHITE),
]
n = len(flow); bw_f = Inches(1.42); gap = Inches(0.2)
total_w = Emu(int(bw_f) * n + int(gap) * (n - 1))
x0 = Emu(int((SW - total_w)) // 2)
y0 = Inches(2.95); bh_f = Inches(0.95)
x = x0
for i, (label, c, tc) in enumerate(flow):
    box = add_rect(s, x, y0, bw_f, bh_f, c, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    tf = box.text_frame; tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.04); tf.margin_right = Inches(0.04)
    tf.margin_top = Inches(0.02); tf.margin_bottom = Inches(0.02)
    for j, ln in enumerate(label.split("\n")):
        p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        R(p, ln, 11, tc, bold=True)
    x = Emu(int(x) + int(bw_f))
    if i < n - 1:
        tb, tf2 = textbox(s, x, y0, gap, bh_f)
        tf2.vertical_anchor = MSO_ANCHOR.MIDDLE
        pp = tf2.paragraphs[0]; pp.alignment = PP_ALIGN.CENTER
        R(pp, "→", 15, FAINT, bold=True)
        x = Emu(int(x) + int(gap))
left = [
    (0, "Data Lake médaillon (Bronze/Silver/Gold, Parquet partitionné year=)", False),
    (0, "→ C1.3 car séparation brut/nettoyé/analytique tracée zone par zone", True),
]
right = [
    (0, "Persistance polyglotte : MySQL (tabulaire) + MongoDB (GeoJSON)", False),
    (0, "→ C1.1 + C1.2 car un type de donnée par moteur, pas un choix par défaut", True),
]
bullets_box(s, left, Inches(0.7), Inches(4.15), Inches(5.95), Inches(1.7), size=13.5)
bullets_box(s, right, Inches(6.85), Inches(4.15), Inches(5.8), Inches(1.7), size=13.5)
notes(s, "Commençons par le besoin. Dix sources de données ouvertes sur Paris, chacune avec son propre "
         "format, et aucune ne répond seule à la vraie question : où vivre ou investir, à quel prix "
         "réel ? Pour gouverner cette donnée éclatée, nous avons construit l'architecture que vous voyez "
         "ici : un data lake en médaillon, qui fait traverser chaque source par trois zones — Bronze "
         "pour le brut, Silver pour le nettoyé, Gold pour l'analytique — partitionné par année. C'est "
         "exactement la compétence C1.3. Ensuite, ces données sont stockées dans deux bases "
         "complémentaires : MySQL pour tout ce qui est tabulaire et filtrable, ça c'est C1.1, et MongoDB "
         "pour les documents géographiques imbriqués, ça c'est C1.2. Nous n'avons pas choisi ces deux "
         "bases par habitude, mais parce que chacune correspond à un type de donnée précis — ce qui "
         "m'amène justement aux choix techniques.")

# --- Slide 7 : 3 · choix techniques — outil retenu vs alternative ---
s = add_slide(); header(s, "Projet 1 · Architecture", "3 · Choix techniques — outil retenu vs alternative", TEAL)
table(s, [
    ["Choix retenu", "Alternative écartée", "Pourquoi", "Compétence"],
    ["Géocodage cascade BAN → BAN Plus (WFS)", "Géocodeur tiers unique",
     "couverture mesurée plus large, pas de point de défaillance unique", "C2.3"],
    ["FastAPI", "Flask / Django REST",
     "typage Pydantic natif, OpenAPI/Swagger auto-généré, I/O asynchrone", "C2.1"],
    ["MySQL + MongoDB", "Tout-SQL ou tout-NoSQL",
     "tabulaire filtrable vs GeoJSON imbriqué : un moteur par type de donnée", "C1.1 · C1.2"],
    ["Parquet partitionné (pyarrow)", "CSV plat",
     "colonnaire, compressé, élagage par partition, interopérable Spark/DuckDB", "C1.3"],
    ["Kafka + Redis Streams (les deux)", "Un seul broker imposé",
     "comparaison robustesse/partitions (Kafka) vs simplicité (Redis)", "C2.2"],
    ["Rate-limiting custom", "Librairie tierce (slowapi…)",
     "zéro dépendance ajoutée, comportement maîtrisé et testé", "C2.1"],
], Inches(1.05), Inches(0.55), Inches(12.25), Inches(5.6),
   col_w=[Inches(3.1), Inches(2.55), Inches(4.8), Inches(1.8)], fs=11.5, accent=TEAL)
notes(s, "Sur cette architecture, chaque outil a été choisi contre une alternative précise, jamais par "
         "défaut. Pour le géocodage, plutôt qu'un service tiers unique, nous avons construit une "
         "cascade : adresse exacte, puis repli sans suffixe, puis un second référentiel géographique — "
         "résultat, une couverture mesurée plus large, et aucun point de défaillance unique. Pour "
         "l'API, FastAPI plutôt que Flask, parce qu'il vérifie les types et génère seul sa "
         "documentation. Pour le data lake, Parquet plutôt qu'un simple CSV, parce qu'il est compressé "
         "et permet de ne lire que l'année demandée. Et pour le streaming, nous avons volontairement "
         "implémenté Kafka et Redis Streams tous les deux, pour comparer en vrai la robustesse de l'un "
         "et la simplicité de l'autre, plutôt que de l'affirmer sans preuve. Justement, parlons preuves.")

# --- Slide 8 : 4 · preuves techniques (sécurité, scalabilité, performance) ---
s = add_slide(); header(s, "Projet 1 · Architecture", "4 · Preuves techniques", TEAL)
bullets_box(s, [
    (0, "Sécurité & API (C2.1) — capture Swagger réelle ci-contre", True),
    (1, "JWT OAuth2 (/auth/token) + rate-limiting (fenêtre glissante, 429 au quota)", False),
    (1, "monitoring live /metrics capturé en direct : 220 req, 38,6 ms moy., 1 lente détectée", False),
    (0, "Scalabilité & résilience (C1.4)", True),
    (1, "nginx + 3 réplicas testés (--scale api=3, /metrics.instance alterne)", False),
    (1, "pool_recycle MySQL (reconnexion auto) ; failover nginx si réplica HS", False),
    (0, "Performance mesurée (C2.4)", True),
    (1, "pipeline ETL : 146,5 s, goulot identifié = persist_storage (86,6 s)", False),
    (1, "API : 8/9 endpoints sous SLA 300 ms ; carte « bâtiment » hors SLA (≈340 ms), diagnostiqué", False),
    (0, "→ C2.1 · C1.4 · C2.4 car sécurité, charge et performance sont mesurées, pas affirmées", True),
], Inches(0.7), Inches(1.5), Inches(5.85), Inches(5.0), size=12.5)
pic(s, "p3_swagger_t.png", Inches(6.85), Inches(1.5), Inches(5.8), Inches(3.55))
img_caption(s, "Swagger /docs — Authorize (JWT), routes verrouillées, /auth/token",
            Inches(6.85), Inches(5.1), Inches(5.8))
tb, tf = textbox(s, Inches(6.85), Inches(5.5), Inches(5.8), Inches(0.5))
R(tf.paragraphs[0], "Détail complet (charts performance, sécurité réseau) → annexes", 11, FAINT, italic=True)
notes(s, "Un choix technique ne vaut rien sans preuve, donc voici trois preuves mesurées aujourd'hui, en "
         "direct. D'abord la sécurité : cette capture d'écran réelle de Swagger montre que l'API est "
         "protégée par un jeton JWT, avec une limite de requêtes par fenêtre de temps — et nous avons "
         "mesuré ce monitoring en envoyant deux cent vingt requêtes de test, pour une latence moyenne de "
         "trente-huit millisecondes. Ensuite la scalabilité : nous avons réellement testé trois réplicas "
         "de l'API derrière un répartiteur de charge, et observé le trafic se répartir entre eux. Enfin "
         "la performance : le pipeline complet prend cent quarante-six secondes, et nous savons "
         "précisément où, l'écriture en base ; côté API, huit endpoints sur neuf respectent notre "
         "objectif de latence, le neuvième est identifié et expliqué. Le détail complet de chacune de "
         "ces mesures est disponible en annexe pour vos questions. Voyons maintenant ce que cette "
         "architecture produit, et où elle s'arrête.")

# --- Slide 9 : 5 · résultats et limites ---
s = add_slide(); header(s, "Projet 1 · Architecture", "5 · Résultats et limites", TEAL)
table(s, [
    ["Indicateur composite", "Sources croisées"],
    ["Score de qualité de vie (/10)", "Bruit + Air (Bruitparif)"],
    ["Mois de salaire pour 1 m²", "Prix DVF + Revenu INSEE"],
    ["Taux d'effort loyer (50 m²)", "Loyer + Revenu INSEE"],
    ["Part de logements sociaux", "INSEE recensement (IRIS)"],
], Inches(1.6), Inches(0.7), Inches(6.2), Inches(1.85),
   col_w=[Inches(3.6), Inches(2.6)], fs=12.5, accent=TEAL)
bullets_box(s, [
    (0, "Résultats", True),
    (1, "indicateurs nés de la fusion — absents de toute source isolée", False),
    (1, "carte MapLibre 4 niveaux (arrondissement → bâtiment) + 8 tests", False),
    (0, "Limites assumées", True),
    (1, "niveau « bâtiment » (~19 000 pts) hors SLA → mesuré, diagnostiqué (slide précédente)", False),
    (1, "déploiement public = perspective", False),
], Inches(0.7), Inches(3.7), Inches(6.5), Inches(3.0), size=14)
pic(s, "p3_section4_t.png", Inches(7.4), Inches(1.5), Inches(5.45), Inches(4.9))
img_caption(s, "Panorama des 20 arrondissements + tendances", Inches(7.4), Inches(6.45), Inches(5.45))
notes(s, "Le vrai résultat de ce projet, ce sont des indicateurs qui n'existent dans aucune source "
         "isolée, parce qu'ils naissent de la fusion : combien de mois de salaire pour un mètre carré, "
         "ou quelle part de logements sociaux à l'échelle d'un quartier. Tout cela est restitué sur une "
         "carte interactive à quatre niveaux de zoom, de l'arrondissement jusqu'au bâtiment, et la "
         "chaîne complète est couverte par huit tests automatisés. Et nous assumons une limite "
         "clairement identifiée : au niveau le plus fin, dix-neuf mille bâtiments, la carte dépasse "
         "notre objectif de latence — nous l'avons mesuré à la slide précédente, diagnostiqué, et la "
         "piste de correction est connue. Ce premier projet ferme ainsi le Bloc 1. Passons maintenant au "
         "Bloc 2, avec la data science : le projet ROI Marketing.")

# ==========================================================================
# PROJET 2 — DATA SCIENCE (BLOC 2) — slides 10 à 14
# ==========================================================================
set_accent(CORAL)
s = divider(
    "2", "ROI Marketing",
    "Système multi-modèles — prédiction & explicabilité des ventes — Bloc 2 (Data Science)",
    CORAL, "BLOC 2 — Compétences démontrées",
    [("C3.1", "Préparation, nettoyage & qualité"),
     ("C3.2", "Dashboard interactif d'aide à la décision"),
     ("C3.3", "Analyse exploratoire & insights"),
     ("C4.1", "Stratégie d'intégration de l'IA"),
     ("C4.2", "Modèle prédictif fonctionnel"),
     ("C4.3", "Évaluation comparative multi-modèles")])
notes(s, "Deuxième projet : ROI Marketing. Il valide la partie data science du Bloc 2 : préparer une "
         "donnée, en tirer des analyses, construire un modèle prédictif, le comparer à d'autres, et "
         "l'intégrer dans une vraie stratégie métier. Ce projet couvre six compétences, de C3.1 à C4.3. "
         "Nous allons suivre la même logique que pour le premier projet : la question métier d'abord, "
         "puis la préparation des données, puis la modélisation, puis l'explicabilité et les limites.")

# --- Slide 11 : question métier & dashboard ---
s = add_slide(); header(s, "Projet 2 · Data Science", "Question métier & dashboard décisionnel (C3.2)", CORAL)
items = [
    (0, "Question métier", True),
    (1, "un CMO doit répartir un budget : TV, Radio, Social, Influenceurs", False),
    (1, "quel canal génère réellement les ventes ? sur quoi investir ?", False),
    (0, "Trois volets indissociables", True),
    (1, "prédire les ventes à partir du mix budgétaire", False),
    (1, "sélectionner objectivement le meilleur modèle", False),
    (1, "expliquer ses décisions de façon fiable", False),
    (0, "Tableau de bord orienté décideur (C3.2)", True),
    (1, "KPIs stratégiques, lisibles, utiles à l'arbitrage", False),
]
split(s, items, image="p2_accueil_t.png", size=14.5,
      cap="Dashboard ROI Marketing — KPIs stratégiques orientés CMO")
notes(s, "La question métier est simple à poser, mais difficile à trancher : un directeur marketing doit "
         "répartir un budget entre quatre canaux — télévision, radio, réseaux sociaux, influenceurs — et "
         "il veut savoir lequel génère vraiment des ventes. Pour y répondre sérieusement, trois volets "
         "sont liés les uns aux autres : prédire les ventes à partir du budget, choisir objectivement le "
         "meilleur modèle plutôt que le premier qui marche, et surtout pouvoir expliquer la décision, "
         "parce qu'un modèle qu'on ne peut pas expliquer ne sera jamais adopté par un comité de "
         "direction. Le tableau de bord à droite démontre la compétence C3.2 : ce n'est pas un graphique "
         "de data scientist, ce sont des indicateurs clés, lisibles par un décideur. Mais avant de "
         "modéliser, il a fallu préparer correctement la donnée.")

# --- Slide 12 : préparation, EDA & anti-fuite ---
s = add_slide(); header(s, "Projet 2 · Data Science", "Préparation, EDA & anti-fuite (C3.1 · C3.3)", CORAL)
items = [
    (0, "Préparation & qualité des données (C3.1)", True),
    (1, "nettoyage, typage, encodage catégoriel (ColumnTransformer)", False),
    (1, "garantie anti-fuite : préprocesseur refit à chaque fold, sur le train seul", False),
    (0, "Analyse exploratoire & insights (C3.3)", True),
    (1, "corrélations canaux/ventes, détection des leviers dominants", False),
    (1, "insight clé : la TV domine très largement les ventes", False),
    (0, "Pipeline de bout en bout & reproductible", True),
    (1, "EDA → préprocessing → modèles → sélection → interprétabilité", False),
]
split(s, items, image="p2_fig_07_correlation_matrix_t.png", size=14.5,
      cap="Matrice de corrélation — analyse exploratoire (C3.3)")
notes(s, "La préparation des données démontre la compétence C3.1 : nettoyage, typage, encodage des "
         "variables catégorielles, via un transformeur de colonnes. Le point méthodologique le plus "
         "important ici, c'est la garantie contre la fuite de données : ce transformeur n'est jamais "
         "ajusté sur l'ensemble du jeu de données, il est recalculé à chaque pli de validation croisée, "
         "uniquement sur les données d'entraînement de ce pli — sinon, nos métriques seraient "
         "artificiellement trop optimistes. L'analyse exploratoire, qui démontre la compétence C3.3, "
         "fait ressortir un insight fort et vérifié, visible sur cette matrice de corrélation : la "
         "télévision domine très largement l'effet sur les ventes. Tout ce travail est encapsulé dans un "
         "pipeline reproductible, de l'analyse jusqu'à l'interprétabilité. La donnée est prête : passons "
         "à la modélisation.")

# --- Slide 13 : modélisation & comparaison ---
s = add_slide(); header(s, "Projet 2 · Data Science", "Modélisation & comparaison multi-modèles (C4.2 · C4.3)", CORAL)
table(s, [
    ["Modèle", "RMSE", "R²", "MAPE"],
    ["XGBoost  ✅", "3,54", "0,9985", "1,84 %"],
    ["Random Forest", "3,70", "0,9984", "1,90 %"],
    ["Rég. linéaire", "5,88", "0,9960", "1,87 %"],
    ["MLP", "5,98", "0,9958", "2,26 %"],
], Inches(1.6), Inches(0.7), Inches(5.7), Inches(2.5),
   col_w=[Inches(2.4), Inches(1.1), Inches(1.2), Inches(1.0)], fs=13, accent=CORAL)
bullets_box(s, [
    (0, "4 modèles comparés (C4.3)", True),
    (1, "baseline linéaire · Random Forest · XGBoost · MLP", False),
    (1, "GridSearchCV 5-fold, métriques adaptées (RMSE/R²/MAPE)", False),
    (0, "Modèle prédictif retenu (C4.2)", True),
    (1, "XGBoost — sélection automatique (RMSE test minimal)", False),
    (1, "écart train/test sain → pas de surapprentissage", False),
], Inches(0.7), Inches(3.6), Inches(6.5), Inches(3.0), size=14)
pic(s, "p2_fig_04_predictions_vs_actual_t.png", Inches(7.4), Inches(1.5), Inches(5.45), Inches(4.9))
img_caption(s, "Prédictions vs réalité (test) — XGBoost retenu", Inches(7.4), Inches(6.45), Inches(5.45))
notes(s, "Quatre modèles sont mis en concurrence dans ce tableau : une régression linéaire comme "
         "référence, une forêt aléatoire, XGBoost, et un réseau de neurones, chacun optimisé en cinq "
         "plis de validation croisée — c'est la compétence C4.3, l'évaluation comparative avec des "
         "métriques adaptées. La sélection du modèle final n'est pas une préférence personnelle, elle "
         "est automatique, sur un seul critère, l'erreur de test la plus faible — c'est la compétence "
         "C4.2. XGBoost l'emporte sur les trois métriques, comme on le voit à droite sur ce graphique "
         "prédictions contre réalité. Une question que le jury pose souvent : un score de zéro virgule "
         "neuf cent quatre-vingt-dix-huit, n'est-ce pas suspect ? Je l'assume : le jeu de données vient "
         "de Kaggle et il est semi-synthétique, mais l'écart entre l'entraînement et le test reste sain, "
         "donc il n'y a pas de surapprentissage. Le modèle prédit bien : reste à savoir s'il est "
         "utilisable en confiance.")

# --- Slide 14 : explicabilité, intégration IA, éco & limites ---
s = add_slide(); header(s, "Projet 2 · Data Science", "Explicabilité, intégration IA & limites (C4.1)", CORAL)
items = [
    (0, "Explicabilité convergente", True),
    (1, "3 méthodes : importance native · permutation · SHAP", False),
    (1, "→ TV > 90 % de l'impact, Influencer < 0,13 %", False),
    (0, "Stratégie d'intégration de l'IA dans le métier (C4.1)", True),
    (1, "API FastAPI = seul accès au modèle (/predict, /explain, /health)", False),
    (1, "dashboard découplé : simulation budgétaire temps réel", False),
    (1, "éco-responsabilité mesurée (≈ 0,5 Wh / 0,03 gCO₂)", False),
    (0, "Limites assumées", True),
    (1, "seuils d'overfitting empiriques · pas de tests unitaires formels", False),
]
split(s, items, image="p2_fig_13_interpretability_comparison_t.png", size=14,
      cap="Convergence des 3 méthodes d'interprétabilité")
notes(s, "Pour qu'un modèle soit utilisable en confiance, il faut pouvoir l'expliquer, pas juste le "
         "croire. Nous avons utilisé trois méthodes mathématiquement indépendantes — l'importance native "
         "du modèle, la permutation, et la méthode SHAP — et les trois arrivent à la même conclusion : "
         "la télévision explique plus de quatre-vingt-dix pour cent de l'effet sur les ventes, les "
         "influenceurs moins de zéro virgule treize pour cent. Ce résultat partagé par trois méthodes "
         "différentes est un signal de robustesse, pas une coïncidence. Pour la stratégie d'intégration "
         "de l'IA dans le métier, la compétence C4.1, le modèle vit derrière une API, jamais chargé "
         "directement par le dashboard, qui permet une simulation budgétaire en temps réel — nous avons "
         "même mesuré son empreinte carbone. Nous assumons deux limites : des seuils de surapprentissage "
         "fixés empiriquement, et l'absence de tests unitaires formels, la validation reposant pour "
         "l'instant sur la validation croisée. Ce deuxième projet ferme la partie data science du Bloc "
         "2. Passons à la deuxième moitié du Bloc 2 : l'intelligence artificielle générative.")

# ==========================================================================
# PROJET 3 — IA GÉNÉRATIVE (BLOC 2) — slides 15 à 17
# ==========================================================================
set_accent(GOLD)
s = divider(
    "3", "AISCA",
    "Analyse sémantique des compétences & recommandation de métiers — Bloc 2 (IA générative)",
    GOLD, "BLOC 2 — Compétences démontrées",
    [("C5.1", "Cas d'usage GenAI pertinent (RH Tech)"),
     ("C5.2", "Solution GenAI fonctionnelle & accessible"),
     ("C5.3", "Évaluation des résultats & ajustement")])
notes(s, "Troisième et dernier projet : AISCA. Il valide la partie IA générative du Bloc 2 : identifier "
         "un cas d'usage pertinent pour cette technologie, développer une solution fonctionnelle et "
         "accessible, puis évaluer et ajuster les résultats qu'elle produit. AISCA recommande des "
         "métiers à un candidat à partir d'un texte libre décrivant ses compétences, et génère une "
         "synthèse personnalisée. Ce projet couvre trois compétences, C5.1 à C5.3. Pour ce dernier bloc, "
         "nous allons aller plus vite : cas d'usage et solution sur une slide, évaluation et valeur "
         "métier sur la suivante.")

# --- Slide 16 : 1+2+3 · cas d'usage, choix du modèle & solution développée ---
s = add_slide(); header(s, "Projet 3 · IA générative", "1+2+3 · Cas d'usage, modèle & solution", GOLD)
items = [
    (0, "Cas d'usage & pourquoi le GenAI (C5.1)", True),
    (1, "deux candidats décrivent la même compétence différemment ; un moteur par mots-clés rate l'équivalence", False),
    (1, "alternatives écartées : synonymes manuels (ne scale pas), classification supervisée (pas de dataset labellisé)", False),
    (0, "Choix du modèle & de l'approche (C5.2)", True),
    (1, "RAG (retrieval SBERT + contexte structuré + génération) plutôt que fine-tuning ou prompting seul", False),
    (1, "Gemini 2.5-flash + cascade de repli (modèle local → template) ; SBERT multilingue", False),
    (0, "Solution développée — architecture & garde-fous", True),
    (1, "Retrieval → Augmented context → Generation ; 1 seul appel API/profil (cache SHA-256)", False),
    (0, "→ C5.1 + C5.2 car cas d'usage justifié et solution robuste, sans dépendance unique", True),
]
split(s, items, image="p1_rag_t.png", size=13, text_w=Inches(6.3),
      cap="Onglet « Agent RAG » — Retrieval · Augmented context · Generation (capture réelle)")
notes(s, "Le cas d'usage part d'un constat RH très concret : deux candidats peuvent décrire la même "
         "compétence avec des mots complètement différents, et un moteur de recherche par mots-clés "
         "rate cette équivalence. Avant de choisir l'IA générative, nous avons écarté deux alternatives "
         "plus simples : un dictionnaire de synonymes, qui ne couvrira jamais tous les cas, et une "
         "classification supervisée, qui demanderait un jeu de données annoté que nous n'avons pas. "
         "C'est la compétence C5.1 : un cas d'usage choisi et justifié, pas par défaut. Notre solution, "
         "qui démontre la compétence C5.2, suit trois étapes visibles sur cette capture d'écran réelle "
         "de l'application : la recherche sémantique du profil par un modèle SBERT multilingue, la "
         "construction d'un contexte structuré avec les scores réels, et enfin la génération par le "
         "modèle Gemini, avec un verrou d'un seul appel par profil et une cascade de secours si le "
         "service cloud est indisponible. Une solution qui fonctionne en théorie ne suffit pas : voyons "
         "si elle fonctionne vraiment, et avec quelle valeur.")

# --- Slide 17 : 4+5 · évaluation, valeur métier & limites ---
s = add_slide(); header(s, "Projet 3 · IA générative", "4+5 · Évaluation, valeur métier & limites", GOLD)
left = [
    (0, "Démo réelle capturée aujourd'hui (C5.2)", True),
    (1, "profil Lina Martin → retrieval réel (BI Analyst 0,76) → bio générée par Gemini (mode=gemini_api)", False),
    (0, "Évaluation & ajustement (C5.3)", True),
    (1, "grille temp×top_p : T=0,0 maximise la qualité (0,82) → réglage retenu en production", False),
    (1, "test automatisé d'absence de biais : classement dépend des compétences, pas du rôle déclaré", False),
    (1, "thinking_budget=0 + budget ×4 → corrige les réponses tronquées de Gemini", False),
    (0, "→ C5.3 car résultats évalués et ajustés sur preuve mesurée, pas au hasard", True),
]
right = [
    (0, "Valeur métier & accessibilité", True),
    (1, "cartographie + recommandation + synthèse en quelques secondes, explicable (scores avant génération)", False),
    (1, "interface Streamlit guidée, profils préchargés, SBERT multilingue", False),
    (1, "limite assumée : l'interface elle-même reste en français uniquement", False),
    (0, "Limites & risques", True),
    (1, "biais du référentiel ; dépendance à Gemini → mitigée par la cascade de repli locale", False),
]
bullets_box(s, left, Inches(0.7), Inches(1.55), Inches(5.95), Inches(5.3), size=13)
bullets_box(s, right, Inches(6.85), Inches(1.55), Inches(5.8), Inches(5.3), size=13)
notes(s, "Pour le prouver, nous avons fait tourner aujourd'hui le pipeline complet sur un vrai profil : "
         "la recherche sémantique a réellement renvoyé le métier de BI Analyst avec un score de zéro "
         "virgule soixante-seize, et Gemini a réellement généré une bio à partir de ce contexte — le "
         "mode affiché, gemini_api, confirme que ce n'est pas un repli ni un modèle local. Côté "
         "évaluation, la compétence C5.3, nous avons testé douze réglages de température et de "
         "diversité, notés par une grille objective, et la température la plus basse maximise "
         "systématiquement la qualité — c'est ce réglage que nous avons retenu en production. Un test "
         "automatisé vérifie même l'absence de biais : le classement des métiers dépend des compétences "
         "réelles, jamais du rôle que le candidat a déclaré viser. La valeur métier est concrète : une "
         "cartographie et une synthèse en quelques secondes, explicables puisque les scores sont "
         "montrés avant la génération, sur une interface web accessible sans installation. Et nous "
         "assumons deux limites : un référentiel de compétences qui peut être biaisé par qui l'a écrit, "
         "et une dépendance à un service externe, mitigée par notre cascade de secours locale. Voilà nos "
         "trois projets : il est temps de prendre du recul sur l'ensemble.")

# ==========================================================================
# SLIDE 18 — SYNTHÈSE DES COMPÉTENCES DÉMONTRÉES
# ==========================================================================
set_accent(TEAL)
s = add_slide(); header(s, "Synthèse", "Compétences démontrées — référentiel RNCP40875", TEAL)
c1 = [
    (0, "Bloc 1 — Projet 1 (Architecture)", True),
    (1, "C1.1 MySQL (séries arrondissement/année)", False),
    (1, "C1.2 MongoDB (GeoJSON)", False),
    (1, "C1.3 Data Lake médaillon", False),
    (1, "C1.4 nginx + 3 réplicas", False),
    (1, "C2.1 API FastAPI sécurisée", False),
    (1, "C2.2 Kafka + Redis Streams", False),
    (1, "C2.3 fusion multi-sources", False),
    (1, "C2.4 CLI idempotente + validation", False),
]
c2 = [
    (0, "Bloc 2 — Projet 2 (Data Science)", True),
    (1, "C3.1 préparation + anti-fuite", False),
    (1, "C3.2 dashboard décisionnel", False),
    (1, "C3.3 EDA & insights (TV dominante)", False),
    (1, "C4.1 intégration IA (API + simu)", False),
    (1, "C4.2 XGBoost retenu", False),
    (1, "C4.3 4 modèles comparés", False),
]
c3 = [
    (0, "Bloc 2 — Projet 3 (IA générative)", True),
    (1, "C5.1 cas d'usage RH Tech", False),
    (1, "C5.2 SBERT + RAG contraint", False),
    (1, "C5.3 évaluation + calibrage", False),
    (0, "Démarche transverse", True),
    (1, "preuve systématique : tests,", False),
    (1, "métriques, validation, mesure", False),
]
bullets_box(s, c1, Inches(0.6), Inches(1.5), Inches(4.25), Inches(5.3), size=13)
bullets_box(s, c2, Inches(4.95), Inches(1.5), Inches(4.25), Inches(5.3), size=13)
bullets_box(s, c3, Inches(9.3), Inches(1.5), Inches(3.8), Inches(5.3), size=13)
notes(s, "Cette slide est la preuve de couverture complète du référentiel. À gauche, le Bloc 1 est "
         "intégralement couvert par le premier projet, de C1.1 à C2.4. Au centre et à droite, le Bloc 2 "
         "est couvert en deux temps : la data science, de C3.1 à C4.3, par le deuxième projet, et l'IA "
         "générative, C5.1 à C5.3, par le troisième. Mais au-delà de cette liste de codes, une démarche "
         "commune relie les trois projets, et c'est elle qui compte le plus : nous ne nous sommes jamais "
         "contentés d'affirmer que quelque chose fonctionne, nous l'avons systématiquement prouvé, par "
         "des tests automatisés, des métriques chiffrées, des validations, et des mesures réelles. Cette "
         "couverture complète ne veut pas dire que tout est parfait : voyons maintenant, honnêtement, ce "
         "qui reste à améliorer.")

# ==========================================================================
# SLIDE 19 — LIMITES ET AMÉLIORATIONS
# ==========================================================================
s = add_slide(); header(s, "Recul critique", "Limites & améliorations", TEAL)
left = [
    (0, "Limites assumées", True),
    (1, "P1 : niveau bâtiment hors SLA · pas de déploiement public", False),
    (1, "P2 : seuils d'overfitting empiriques · pas de tests formels", False),
    (1, "P3 : biais possible du référentiel · dépendance modèle externe", False),
]
right = [
    (0, "Améliorations identifiées", True),
    (1, "P1 : tuilage vectoriel · planificateur réel · déploiement", False),
    (1, "P2 : feature engineering · suite de tests · monitoring", False),
    (1, "P3 : fine-tuning SBERT · base SQL · calibrage formel du seuil", False),
    (0, "Transverse : MLOps complet", True),
    (1, "CI/CD, versionnage de modèles, observabilité", False),
]
bullets_box(s, left, Inches(0.7), Inches(1.6), Inches(6.0), Inches(5), size=15)
bullets_box(s, right, Inches(6.9), Inches(1.6), Inches(6.0), Inches(5), size=15)
notes(s, "Reconnaître une limite n'est pas un aveu d'échec, c'est au contraire une preuve de recul "
         "professionnel, et nous préférons l'assumer plutôt que de la cacher. Chaque projet a la "
         "sienne, et en face, une piste d'amélioration déjà identifiée : pour l'architecture, le "
         "tuilage vectoriel et un vrai déploiement public ; pour la data science, des tests automatisés "
         "plus complets et un monitoring en production ; pour l'IA générative, un calibrage plus formel "
         "du seuil de décision. Et de façon transverse aux trois projets, notre prochaine étape commune "
         "serait une démarche MLOps complète, avec intégration continue, suivi de version des modèles, "
         "et observabilité — pour passer d'un prototype solide à un système réellement opéré en "
         "production. Concluons maintenant sur ce que ces trois projets démontrent ensemble.")

# ==========================================================================
# SLIDE 20 — CONCLUSION TRANSVERSALE
# ==========================================================================
s = add_slide()
add_rect(s, 0, 0, SW, SH, NAVY)
add_rect(s, 0, 0, SW, Inches(0.16), TEAL)
add_rect(s, 0, Inches(7.34), SW, Inches(0.16), GOLD)
add_rect(s, Inches(0.9), Inches(0.85), Inches(2.4), Pt(4), GOLD)
tb, tf = textbox(s, Inches(0.9), Inches(1.0), Inches(11.4), Inches(0.7))
R(tf.paragraphs[0], "CONCLUSION TRANSVERSALE", 16, GOLD, bold=True)
bullets_box_items = [
    (0, "Trois projets, deux blocs, une même exigence d'ingénierie", True),
    (1, "Bloc 1 : architecture polyglotte fiable, intégrée et scalable (Projet 1)", False),
    (1, "Bloc 2 : data science rigoureuse & explicable (Projet 2)", False),
    (1, "Bloc 2 : IA générative maîtrisée et évaluée (Projet 3)", False),
    (0, "Apports professionnels", True),
    (1, "réflexe d'architecture : séparer les responsabilités", False),
    (1, "réflexe de preuve : tests, métriques, validation, mesure", False),
    (1, "capacité à assumer les limites et à proposer des pistes", False),
]
tb, tf = textbox(s, Inches(0.9), Inches(1.75), Inches(11.4), Inches(3.6))
first = True
for level, text, bold in bullets_box_items:
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    first = False
    p.space_after = Pt(6)
    if level == 0:
        R(p, "▸ ", 18, GOLD, bold=True)
        R(p, text, 18, WHITE, bold=True)
    else:
        R(p, "•  ", 15, GOLD)
        R(p, text, 15, RGBColor(0xD5, 0xDE, 0xEA))
tb, tf = textbox(s, Inches(0.9), Inches(5.55), Inches(11.4), Inches(1.0))
R(tf.paragraphs[0],
  "« Du texte au chiffre, du chiffre à la carte — toujours de façon robuste, explicable et reproductible. »",
  18, GOLD, italic=True, font=FONT_H)
tb, tf = textbox(s, Inches(0.9), Inches(6.5), Inches(11.4), Inches(0.6))
R(tf.paragraphs[0], "Merci de votre attention — Questions & échanges", 16, WHITE, bold=True)
notes(s, "Pour conclure, ces trois projets sont des preuves complémentaires, pas redondantes, de la "
         "maîtrise des Blocs 1 et 2. Le projet d'architecture démontre que nous savons concevoir et "
         "superviser un système de données fiable. Le projet de data science démontre que nous savons "
         "en tirer un modèle prédictif rigoureux et explicable. Le projet d'IA générative démontre que "
         "nous savons identifier un bon cas d'usage et évaluer honnêtement ce qu'il produit. Deux "
         "réflexes nous relient sur les trois projets : penser l'architecture avant le code, et "
         "toujours préférer la preuve à l'affirmation. Merci de votre attention, nous sommes maintenant "
         "prêts pour vos questions.\n\n"
         "Anticipation des questions du jury :\n"
         "• Pourquoi MySQL + MongoDB ? → polyglot persistence : relationnel pour le tabulaire filtrable, "
         "documentaire pour le GeoJSON.\n"
         "• Comment l'architecture est-elle scalable / résiliente ? → docker-compose + nginx, 3 réplicas, "
         "streaming Kafka partitionné.\n"
         "• Comment mesurez-vous la performance des pipelines ? → CLI idempotente, validation Gold, "
         "latence par niveau cartographique.\n"
         "• Le R² de 0,998 n'est-il pas suspect ? → dataset semi-synthétique ; écart train/test sain.\n"
         "• Comment garantir l'absence de fuite / biais ? → pipeline refit par fold (P2) ; test dédié (P3).\n"
         "• Pourquoi l'IA générative ici ? → mesurer le sens d'un texte libre, impossible par mots-clés.")

out = Path(os.environ.get("PPTX_OUT", Path(__file__).resolve().parents[1] / "Soutenance_3_Projets.pptx"))
prs.save(str(out))
print(f"OK -> {out}  ({len(prs.slides._sldIdLst)} slides)")
