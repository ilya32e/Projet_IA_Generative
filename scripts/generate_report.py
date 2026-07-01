"""Genere un rapport PDF detaille du projet AISCA pour la soutenance.

Usage : python scripts/generate_report.py
Sortie : AISCA_Rapport_Soutenance.pdf (a la racine du projet)
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

PRIMARY = colors.HexColor("#12395a")
ACCENT = colors.HexColor("#1f7a8c")
LIGHT = colors.HexColor("#eef3f7")
DARK = colors.HexColor("#13293d")
GREY = colors.HexColor("#5b6b7a")

AUTHORS = "Binome : Salah-Eddine Zbiri  -  Ilyasse Mouradi"
DATE_STR = datetime.now().strftime("%d/%m/%Y")
OUTPUT = ROOT / "AISCA_Rapport_Soutenance.pdf"


# --------------------------------------------------------------------------- #
# Styles
# --------------------------------------------------------------------------- #
def build_styles():
    ss = getSampleStyleSheet()
    styles = {}
    styles["title"] = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                                      fontSize=26, textColor=PRIMARY, leading=30, alignment=TA_CENTER)
    styles["subtitle"] = ParagraphStyle("subtitle", parent=ss["Normal"], fontSize=13, textColor=ACCENT,
                                         leading=18, alignment=TA_CENTER, spaceBefore=6)
    styles["covermeta"] = ParagraphStyle("covermeta", parent=ss["Normal"], fontSize=11, textColor=DARK,
                                          leading=18, alignment=TA_CENTER)
    styles["h1"] = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15,
                                   textColor=colors.white, leading=20, spaceBefore=16, spaceAfter=10,
                                   backColor=PRIMARY, borderPadding=(6, 8, 6, 8), leftIndent=0)
    styles["h2"] = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12.5,
                                   textColor=PRIMARY, leading=16, spaceBefore=12, spaceAfter=5)
    styles["h3"] = ParagraphStyle("h3", parent=ss["Heading3"], fontName="Helvetica-Bold", fontSize=11,
                                   textColor=ACCENT, leading=14, spaceBefore=8, spaceAfter=3)
    styles["body"] = ParagraphStyle("body", parent=ss["Normal"], fontSize=10.3, textColor=DARK,
                                     leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
    styles["bullet"] = ParagraphStyle("bullet", parent=styles["body"], leftIndent=14, bulletIndent=4, spaceAfter=3)
    styles["caption"] = ParagraphStyle("caption", parent=ss["Normal"], fontSize=8.6, textColor=GREY,
                                        leading=11, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
    styles["code"] = ParagraphStyle("code", parent=ss["Normal"], fontName="Courier", fontSize=9,
                                     textColor=DARK, backColor=LIGHT, leading=12, borderPadding=(6, 6, 6, 6),
                                     spaceAfter=8)
    styles["cell"] = ParagraphStyle("cell", parent=ss["Normal"], fontSize=8.8, textColor=DARK, leading=12)
    styles["cellb"] = ParagraphStyle("cellb", parent=styles["cell"], fontName="Helvetica-Bold")
    styles["cellh"] = ParagraphStyle("cellh", parent=styles["cell"], fontName="Helvetica-Bold", textColor=colors.white)
    styles["toc"] = ParagraphStyle("toc", parent=ss["Normal"], fontSize=11, textColor=DARK, leading=20)
    return styles


S = build_styles()


def esc(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def P(text, style="body"):
    return Paragraph(esc(text) if style != "raw" else text, S["body"] if style == "raw" else S[style])


def H1(text):
    return Paragraph(esc(text), S["h1"])


def H2(text):
    return Paragraph(esc(text), S["h2"])


def H3(text):
    return Paragraph(esc(text), S["h3"])


def bullets(items):
    return [Paragraph("&bull;&nbsp;&nbsp;" + esc(it), S["bullet"]) for it in items]


def code(text):
    return Paragraph(esc(text).replace("\n", "<br/>").replace(" ", "&nbsp;"), S["code"])


def make_table(header, rows, col_widths, header_bg=PRIMARY):
    data = [[Paragraph(esc(h), S["cellh"]) for h in header]]
    for r in rows:
        data.append([c if isinstance(c, Paragraph) else Paragraph(esc(str(c)), S["cell"]) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, PRIMARY),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cdd8e2")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
    ]
    t.setStyle(TableStyle(style))
    return t


# --------------------------------------------------------------------------- #
# Donnees reelles du projet
# --------------------------------------------------------------------------- #
def load_project_data():
    from src.analytics import reference_kpis
    from src.data_pipeline import load_reference_data, load_sample_profiles, prepare_all_data
    prepare_all_data()
    ref, jobs = load_reference_data()
    kpis = reference_kpis(ref, jobs)
    example = None
    try:
        from src.recommender import analyse_submission
        from src.semantic_engine import SemanticEngine
        eng = SemanticEngine(backend="auto")
        res = analyse_submission(load_sample_profiles()[1], ref, jobs, eng)
        example = {
            "name": load_sample_profiles()[1]["candidate_name"],
            "backend": eng.info().backend,
            "blocks": res["block_scores"][["block_name", "block_score", "score_label"]].values.tolist(),
            "top": res["top_jobs"][["job_title", "final_score", "score_label"]].values.tolist(),
            "score": res["final_score"],
        }
    except Exception as exc:  # pragma: no cover
        print("Exemple de scoring indisponible:", exc)
    return ref, jobs, kpis, example


# --------------------------------------------------------------------------- #
# Mise en page (numeros de page + bandeau)
# --------------------------------------------------------------------------- #
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(2 * cm, 1.1 * cm, "AISCA - Rapport de projet")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, "Page %d" % doc.page)
    canvas.setStrokeColor(colors.HexColor("#cdd8e2"))
    canvas.setLineWidth(0.4)
    canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.restoreState()


def on_cover(canvas, canvas_doc):
    canvas.saveState()
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, A4[1] - 4.2 * cm, A4[0], 4.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, A4[1] - 4.35 * cm, A4[0], 0.15 * cm, fill=1, stroke=0)
    canvas.setFillColor(PRIMARY)
    canvas.rect(0, 0, A4[0], 1.2 * cm, fill=1, stroke=0)
    canvas.restoreState()


# --------------------------------------------------------------------------- #
# Contenu
# --------------------------------------------------------------------------- #
def build_story(ref, jobs, kpis, example):
    story = []

    # ---------- Page de garde ----------
    story.append(Spacer(1, 5.2 * cm))
    story.append(Paragraph("AISCA", S["title"]))
    story.append(Paragraph("Agent Intelligent Semantique et Generatif<br/>pour la Cartographie des Competences", S["subtitle"]))
    story.append(Spacer(1, 1.0 * cm))
    story.append(Paragraph("Analyse Semantique pour la Cartographie des Competences<br/>et la Recommandation de Metiers", S["covermeta"]))
    story.append(Spacer(1, 1.6 * cm))
    story.append(Paragraph("Projet IA Generative - NLP Semantique &amp; RAG", S["covermeta"]))
    story.append(Paragraph("RNCP40875 - Expert en ingenierie de donnees - Bloc 2", S["covermeta"]))
    story.append(Spacer(1, 2.2 * cm))
    story.append(Paragraph(AUTHORS, S["covermeta"]))
    story.append(Paragraph("EFREI - Data Engineering &amp; AI - 2025/2026", S["covermeta"]))
    story.append(Paragraph("Date : " + DATE_STR, S["covermeta"]))
    story.append(NextPageTemplate("main"))
    story.append(PageBreak())

    # ---------- Sommaire ----------
    story.append(H1("Sommaire"))
    toc = [
        "1.  Introduction et contexte",
        "2.  Objectifs du projet",
        "3.  Cadre theorique",
        "4.  Architecture generale du systeme",
        "5.  Acquisition des donnees (EF1)",
        "6.  Referentiel de competences et de metiers (EF2.1)",
        "7.  Moteur semantique : SBERT et similarite cosinus (EF2.2 - EF2.3)",
        "8.  Systeme de scoring et de recommandation (EF3)",
        "9.  Agent RAG et IA generative controlee (EF4)",
        "10. Interface utilisateur (Streamlit)",
        "11. Conformite aux exigences fonctionnelles",
        "12. Competences RNCP - Bloc 2",
        "13. Tests et validation",
        "14. Choix techniques et justifications",
        "15. Limites et perspectives",
        "16. Conclusion",
        "Annexes : structure du projet, stack et commandes",
    ]
    for item in toc:
        story.append(Paragraph(esc(item), S["toc"]))
    story.append(PageBreak())

    # ---------- 1. Introduction ----------
    story.append(H1("1. Introduction et contexte"))
    story.append(P(
        "AISCA (Agent Intelligent Semantique et Generatif pour la Cartographie des Competences) est une "
        "application web qui aide un utilisateur a evaluer ses competences a partir de reponses en langage "
        "naturel, puis a decouvrir les metiers qui correspondent le mieux a son profil. Le projet s'inscrit "
        "dans le domaine du Traitement Automatique du Langage Naturel (NLP) et de l'IA Generative."))
    story.append(P(
        "Contrairement aux approches statistiques classiques (TF-IDF, sac de mots) qui se limitent a compter "
        "des occurrences, AISCA realise une analyse semantique : il compare le sens des reponses de "
        "l'utilisateur a un referentiel structure de competences, en s'appuyant sur des representations "
        "vectorielles contextuelles (embeddings). Le systeme determine quelles competences sont couvertes, "
        "a quel degre, et quels metiers correspondent au profil analyse."))
    story.append(H2("Probleme metier"))
    story.append(P(
        "L'orientation professionnelle et la cartographie des competences sont des enjeux majeurs en HRTech "
        "et EdTech. Un candidat decrit souvent ses competences avec ses propres mots, sans utiliser le "
        "vocabulaire exact d'un referentiel. Une simple recherche par mots-cles echoue alors a reconnaitre "
        "des competences pourtant maitrisees. L'analyse semantique resout ce probleme en mesurant la "
        "proximite de sens plutot que la correspondance litterale."))

    # ---------- 2. Objectifs ----------
    story.append(H1("2. Objectifs du projet"))
    story.append(P("Le projet vise a construire un pipeline complet, de la collecte de donnees jusqu'a la generation de recommandations :"))
    story.extend(bullets([
        "Collecter les competences via un questionnaire hybride (texte libre + auto-evaluation).",
        "Comparer semantiquement ces reponses a un referentiel de competences avec un modele SBERT local.",
        "Calculer un score de couverture pondere par bloc de competences.",
        "Recommander les 3 metiers les plus pertinents pour le profil.",
        "Generer, de facon controlee, un plan de progression et une bio professionnelle (IA generative).",
        "Respecter les contraintes de cout (Free Tier) via un cache et une limitation stricte des appels API.",
    ]))
    story.append(P(
        "L'ensemble constitue un mini-agent RAG (Retrieval-Augmented Generation) specialise dans l'analyse "
        "de competences : la generation reste guidee par des donnees recuperees et structurees, ce qui "
        "garantit fiabilite, coherence et controle des couts."))

    # ---------- 3. Cadre theorique ----------
    story.append(H1("3. Cadre theorique"))
    story.append(H2("3.1 Analyse semantique"))
    story.append(P(
        "L'analyse semantique vise a comprendre le sens reel des mots, phrases et paragraphes. Elle depasse "
        "les representations numeriques simples, capture les relations conceptuelles et prend en compte le "
        "contexte. Dans AISCA, elle sert a comparer des reponses libres a un referentiel structure."))
    story.append(H2("3.2 Embeddings et SBERT"))
    story.append(P(
        "Un embedding est un vecteur numerique qui represente le sens d'un texte. Les modeles contextuels "
        "(BERT, RoBERTa) produisent des representations sensibles au contexte. SBERT (Sentence-BERT) est une "
        "variante optimisee pour encoder des phrases entieres en vecteurs comparables directement. AISCA "
        "utilise un modele SBERT multilingue, execute localement, donc a cout zero et sans dependance reseau."))
    story.append(H2("3.3 Similarite cosinus"))
    story.append(P(
        "La similarite cosinus mesure l'angle entre deux vecteurs : proche de 1, les textes ont un sens "
        "tres proche ; proche de 0, ils sont sans rapport. C'est la mesure utilisee pour comparer chaque "
        "preuve utilisateur a chaque phrase de competence du referentiel."))
    story.append(code("cos(u, v) = (u . v) / (||u|| x ||v||)"))
    story.append(H2("3.4 IA generative et LLM"))
    story.append(P(
        "Les LLM (Large Language Models) sont entraines sur de vastes corpus et savent comprendre des "
        "instructions, raisonner et rediger. Dans AISCA, l'IA generative est utilisee de facon ciblee et "
        "limitee : enrichir une reponse trop courte, generer un plan de progression et produire une bio."))
    story.append(H2("3.5 Architecture RAG"))
    story.append(P(
        "Un agent RAG combine la recherche d'information avec la generation IA. Il se decompose en trois "
        "etapes : Retrieval (recuperer les elements pertinents), Augmented context (construire un contexte "
        "structure), Generation (produire le contenu final). Cette approche reduit les hallucinations et "
        "maintient la generation alignee avec les donnees du referentiel."))

    # ---------- 4. Architecture ----------
    story.append(H1("4. Architecture generale du systeme"))
    story.append(P("Le systeme suit un pipeline lineaire en quatre grandes etapes, chacune implementee dans un module Python dedie :"))
    story.append(code(
        "Questionnaire (Streamlit)\n"
        "      |\n"
        "      v\n"
        "[1] data_pipeline.py    -> normalisation, construction des preuves, stockage JSON/CSV\n"
        "      |\n"
        "      v\n"
        "[2] semantic_engine.py  -> embeddings SBERT + similarite cosinus (repli TF-IDF)\n"
        "      |\n"
        "      v\n"
        "[3] recommender.py      -> scoring par competence/bloc, classement des metiers (top 3)\n"
        "      |\n"
        "      v\n"
        "[4] genai.py            -> agent RAG : enrichissement, plan, bio (cache + fallback)"))
    story.append(Paragraph("Figure 1 - Pipeline NLP / Scoring / Recommandation / GenAI", S["caption"]))
    story.append(make_table(
        ["Module", "Role"],
        [
            ["config.py", "Configuration, chemins, parametres (modeles, seuil, top N)."],
            ["data_pipeline.py", "Preparation et nettoyage du referentiel, construction des preuves utilisateur, stockage structure."],
            ["semantic_engine.py", "Encodage semantique (SBERT) et calcul de la similarite cosinus, avec repli lexical TF-IDF."],
            ["recommender.py", "Scoring des competences, agregation par bloc, scoring et classement des metiers."],
            ["genai.py", "Agent RAG : enrichissement, plan, bio ; cache local, journalisation, fallback de provider."],
            ["analytics.py", "Indicateurs, heatmap des profils, evaluation qualite des textes, historique des appels."],
        ],
        [4.2 * cm, 11.3 * cm]))

    # ---------- 5. Acquisition ----------
    story.append(H1("5. Acquisition des donnees (EF1)"))
    story.append(H2("5.1 Questionnaire hybride (EF1.1)"))
    story.append(P(
        "Le questionnaire ne se limite pas a des valeurs numeriques : il combine plusieurs types de "
        "questions afin de collecter des preuves contextuelles exploitables par l'analyse semantique :"))
    story.extend(bullets([
        "Texte libre : description de projets, dashboards et usages de l'IA generative.",
        "Echelle de Likert (0-5) : auto-evaluation des niveaux (Python, visualisation, EDA, NLP, GenAI).",
        "Question guidee : utilisation des techniques de tokenization.",
        "Choix multiples : outils utilises et blocs de competences a mettre en avant.",
    ]))
    story.append(P(
        "Pour chaque profil, le module data_pipeline transforme ces reponses en une liste de phrases-preuves "
        "(par exemple, les niveaux Likert sont convertis en phrases descriptives), ce qui ameliore la "
        "qualite des embeddings et de l'appariement semantique."))
    story.append(H2("5.2 Stockage structure (EF1.2)"))
    story.append(P(
        "Chaque soumission recoit un identifiant stable (hash du contenu) et est enregistree en JSON "
        "individuel, puis indexee dans un fichier CSV agrege. Le format structure permet la tracabilite et "
        "le retraitement ulterieur."))

    # ---------- 6. Referentiel ----------
    story.append(H1("6. Referentiel de competences et de metiers (EF2.1)"))
    story.append(P(
        "Le referentiel s'inspire de standards professionnels (ROME, e-Competence Framework). Il comprend "
        "%d competences reparties en %d blocs, et %d profils de metiers. Chaque competence est exprimee sous "
        "forme d'une courte phrase, ce qui la rend directement comparable aux preuves utilisateur."
        % (kpis["competencies"], kpis["blocks"], kpis["jobs"])))
    story.append(H2("6.1 Blocs de competences"))
    block_rows = []
    for bid, bname in sorted(set(zip(ref["block_id"], ref["block_name"]))):
        count = int((ref["block_id"] == bid).sum())
        block_rows.append([bid, bname, str(count)])
    story.append(make_table(["Bloc", "Intitule", "Nb competences"], block_rows, [2.2 * cm, 9.8 * cm, 3.5 * cm]))
    story.append(Spacer(1, 4 * mm))
    story.append(H2("6.2 Profils de metiers"))
    job_rows = [[j.job_id, j.job_title, Paragraph(esc(j.description), S["cell"])] for j in jobs.itertuples()]
    story.append(make_table(["ID", "Metier", "Description"], job_rows, [1.4 * cm, 3.9 * cm, 10.2 * cm]))
    story.append(Paragraph(
        "Chaque metier est defini par un ensemble de competences requises et de blocs prioritaires, "
        "ce qui permet de calculer un taux de couverture par metier.", S["caption"]))

    # ---------- 7. Moteur semantique ----------
    story.append(H1("7. Moteur semantique : SBERT et similarite cosinus (EF2.2 - EF2.3)"))
    story.append(P(
        "Le moteur (semantic_engine.py) encode les preuves utilisateur et les phrases du referentiel en "
        "vecteurs avec un modele SBERT multilingue, puis calcule la similarite cosinus entre chaque preuve "
        "et chaque competence. Les embeddings sont normalises, ce qui ramene le produit scalaire a la "
        "similarite cosinus, bornee dans l'intervalle [0, 1]."))
    story.append(H2("7.1 Robustesse : repli lexical"))
    story.append(P(
        "Si le modele SBERT ne peut pas etre charge (absence de reseau ou de dependances), le moteur bascule "
        "automatiquement sur un repli lexical TF-IDF avec bi-grammes. L'application reste donc fonctionnelle "
        "en toutes circonstances - un choix d'ingenierie important pour une demonstration fiable."))
    story.append(code(
        "left  = model.encode(preuves, normalize_embeddings=True)\n"
        "right = model.encode(competences, normalize_embeddings=True)\n"
        "similarite = clip(left @ right.T, 0, 1)   # cosinus borne [0,1]"))

    # ---------- 8. Scoring ----------
    story.append(H1("8. Systeme de scoring et de recommandation (EF3)"))
    story.append(H2("8.1 Score par competence"))
    story.append(P(
        "Pour chaque competence du referentiel, on retient la similarite maximale obtenue parmi toutes les "
        "preuves de l'utilisateur. C'est le degre auquel le profil couvre cette competence."))
    story.append(H2("8.2 Agregation par bloc"))
    story.append(P(
        "Les scores des competences d'un meme bloc sont moyennes (moyenne ponderee par l'importance de "
        "chaque competence) pour obtenir un score de couverture par bloc, illustre par le graphique radar."))
    story.append(code("score_bloc = somme(Wi x Si) / somme(Wi)"))
    story.append(H2("8.3 Score par metier et recommandation (EF3.2)"))
    story.append(P(
        "Pour chaque metier, on considere ses competences requises et l'on calcule deux indicateurs : la "
        "similarite moyenne sur ces competences, et le taux de couverture (proportion de competences "
        "requises dont la similarite depasse le seuil). Le score final combine ces deux signaux :"))
    story.append(code("score_metier = 0.60 x similarite_moyenne + 0.40 x taux_de_couverture"))
    story.append(P(
        "Les metiers sont ensuite classes par score decroissant et le systeme retourne le Top 3. Ce choix "
        "de formule resulte d'une calibration : la similarite moyenne mesure la qualite de correspondance, "
        "tandis que le taux de couverture recompense l'etendue reelle des competences maitrisees. La "
        "recommandation suit ainsi fidelement le profil saisi, independamment du role declare par l'utilisateur."))
    if example:
        story.append(H2("8.4 Exemple chiffre"))
        story.append(P(
            "Exemple reel calcule par le moteur (%s) pour le profil d'exemple \"%s\". Score global de "
            "couverture : %.2f." % (example["backend"].upper(), example["name"], example["score"])))
        story.append(make_table(
            ["Bloc de competences", "Score", "Niveau"],
            [[b[0], "%.2f" % b[1], b[2]] for b in example["blocks"]],
            [9.8 * cm, 2.8 * cm, 2.9 * cm]))
        story.append(Spacer(1, 3 * mm))
        story.append(make_table(
            ["Metier recommande (Top 3)", "Score final", "Niveau"],
            [[t[0], "%.2f" % t[1], t[2]] for t in example["top"]],
            [9.8 * cm, 2.8 * cm, 2.9 * cm], header_bg=ACCENT))

    # ---------- 9. RAG / GenAI ----------
    story.append(H1("9. Agent RAG et IA generative controlee (EF4)"))
    story.append(P("La couche generative est structuree en agent RAG et volontairement contrainte pour respecter le Free Tier."))
    story.append(H2("9.1 Les trois etapes RAG"))
    story.extend(bullets([
        "Retrieval : recuperation des points forts du profil, des competences faibles (gaps) et des ecarts vis-a-vis du metier cible.",
        "Augmented context : construction d'un contexte structure (scores, forces, gaps, metiers) transmis au LLM.",
        "Generation : production du plan de progression et de la bio a partir de ce contexte uniquement.",
    ]))
    story.append(H2("9.2 Fonctions GenAI"))
    story.extend(bullets([
        "EF4.1 - Enrichissement conditionnel : si les reponses sont trop courtes (< 18 mots), la GenAI les reformule legerement avant l'analyse, ce qui ameliore la precision des embeddings.",
        "EF4.2 - Plan de progression : identifie les competences prioritaires (scores les plus faibles) et propose un parcours d'apprentissage, en un seul appel API.",
        "EF4.3 - Bio professionnelle : courte synthese de type executive summary, en un seul appel API.",
    ]))
    story.append(H2("9.3 Contraintes Free Tier"))
    story.append(make_table(
        ["Contrainte", "Mise en oeuvre"],
        [
            ["Appels API limites", "Un seul appel logique par plan et par bio (verrou generate_once)."],
            ["Caching automatique", "Reponses stockees dans cache/genai_cache.json ; requete identique reutilisee."],
            ["Tracabilite", "Journal des appels (mode, cache, longueur) dans cache/genai_log.csv."],
            ["Repli robuste", "Cascade Gemini -> modele local -> template ; l'app fonctionne sans cle API."],
        ],
        [4.6 * cm, 10.9 * cm]))
    story.append(H2("9.4 Optimisation specifique au modele"))
    story.append(P(
        "Le modele Gemini 2.5 Flash est un modele a raisonnement (thinking) : les tokens de raisonnement "
        "sont decomptes du budget de sortie. Avec une limite de tokens trop basse, la reponse etait tronquee. "
        "La solution mise en place desactive le raisonnement (thinking_budget = 0) et reserve un plancher de "
        "tokens suffisant, garantissant des sorties completes et de qualite tout en restant economique."))

    # ---------- 10. Interface ----------
    story.append(H1("10. Interface utilisateur (Streamlit)"))
    story.append(P("L'interface est organisee en cinq onglets et une barre laterale d'etat du systeme, pensee pour la demonstration :"))
    story.append(make_table(
        ["Onglet", "Contenu"],
        [
            ["Questionnaire", "Formulaire hybride, chargement de profils d'exemple, option d'enrichissement GenAI."],
            ["Resultats", "Score global, radar par bloc, Top 3, classement complet, gaps, export du rapport."],
            ["Agent RAG", "Visualisation des 3 etapes RAG, generation du plan et de la bio, metriques de qualite."],
            ["Analytics", "Heatmap des profils d'exemple, journal des appels GenAI, soumissions enregistrees."],
            ["Methodologie", "Pipeline, formules et tableau de conformite aux exigences fonctionnelles."],
        ],
        [3.5 * cm, 12.0 * cm]))
    story.append(P(
        "La barre laterale affiche en temps reel l'etat du moteur semantique (SBERT ou repli), le statut "
        "GenAI (Gemini actif ou repli), le nombre d'entrees en cache et les indicateurs du referentiel."))

    # ---------- 11. Conformite EF ----------
    story.append(H1("11. Conformite aux exigences fonctionnelles"))
    story.append(make_table(
        ["Exigence", "Intitule", "Implementation"],
        [
            ["EF1.1", "Questionnaire hybride", "Texte libre + Likert + question guidee + choix multiples."],
            ["EF1.2", "Stockage structure", "JSON par soumission + index CSV."],
            ["EF2.1", "Referentiel", "%d competences, %d blocs, %d metiers (inspire ROME / e-CF)." % (kpis["competencies"], kpis["blocks"], kpis["jobs"])],
            ["EF2.2", "Embeddings SBERT locaux", "SentenceTransformer multilingue, execution locale."],
            ["EF2.3", "Similarite cosinus", "Produit scalaire d'embeddings normalises, borne [0,1]."],
            ["EF3.1", "Formule de score ponderee", "0.60 x moyenne + 0.40 x couverture ; seuil configurable."],
            ["EF3.2", "Top 3 metiers", "Classement decroissant et restitution des 3 meilleurs."],
            ["EF4.1", "Enrichissement conditionnel", "Reformulation GenAI si reponses < 18 mots."],
            ["EF4.2", "Plan de progression", "Un appel verrouille + cache."],
            ["EF4.3", "Bio professionnelle", "Un appel verrouille + cache."],
        ],
        [1.8 * cm, 4.6 * cm, 9.1 * cm]))

    # ---------- 12. RNCP ----------
    story.append(H1("12. Competences RNCP - Bloc 2"))
    story.append(P("Le projet permet de valider plusieurs competences du Bloc 2 (BC2 - Piloter et implementer des solutions d'IA) :"))
    story.append(make_table(
        ["Competence visee", "Demonstration dans le projet"],
        [
            ["Collecter et preparer des donnees", "Questionnaire hybride, nettoyage et structuration du referentiel."],
            ["Concevoir des modeles NLP / IA", "Moteur d'embeddings SBERT et appariement semantique."],
            ["Evaluer et optimiser les modeles", "Seuil calibre, metriques de qualite, optimisation du modele Gemini."],
            ["Prototyper des solutions IA", "API, NLP, RAG, embeddings et GenAI integres dans un MVP."],
            ["Developper un pipeline de bout en bout", "Acquisition -> NLP -> scoring -> recommandation -> generation."],
            ["Industrialiser une solution", "Architecture modulaire, cache, fallback, contraintes de cout."],
            ["Documenter et presenter", "Documentation technique, rapport et interface de demonstration."],
        ],
        [5.6 * cm, 9.9 * cm], header_bg=ACCENT))

    # ---------- 13. Tests ----------
    story.append(H1("13. Tests et validation"))
    story.append(P("Une suite de tests automatises valide les comportements critiques du pipeline :"))
    story.extend(bullets([
        "Coherence de la preparation des donnees (absence de doublons apres nettoyage).",
        "Recommandation : le systeme retourne bien un Top 3 et un score borne dans [0, 1].",
        "Stabilite de l'identifiant de soumission pour un meme contenu.",
        "Reutilisation effective du cache GenAI (deuxieme appel servi depuis le cache).",
        "Verrou d'un seul appel par profil pour le plan de progression.",
        "Independance du classement vis-a-vis du role declare par l'utilisateur.",
    ]))
    story.append(P("L'ensemble de la suite (6 tests) passe avec succes, ce qui garantit la non-regression lors des evolutions."))

    # ---------- 14. Choix techniques ----------
    story.append(H1("14. Choix techniques et justifications"))
    story.append(make_table(
        ["Choix", "Justification"],
        [
            ["SBERT local", "Cout zero, pas de dependance reseau pour l'analyse, conforme a l'exigence Open-Source."],
            ["Repli TF-IDF", "Garantit une demonstration fonctionnelle meme sans modele lourd."],
            ["Streamlit", "Prototypage rapide d'une interface data interactive en Python pur."],
            ["Gemini 2.5 Flash", "API GenAI moderne, rapide et gratuite (Free Tier), avec repli local."],
            ["Cache JSON local", "Reduit les appels API, accelere les reponses, respecte le Free Tier."],
            ["Architecture modulaire", "Separation claire des responsabilites, testabilite et maintenance facilitees."],
        ],
        [4.0 * cm, 11.5 * cm]))

    # ---------- 15. Limites ----------
    story.append(H1("15. Limites et perspectives"))
    story.append(H3("Limites actuelles"))
    story.extend(bullets([
        "Referentiel volontairement compact (a vocation pedagogique) ; un deploiement reel necessiterait un referentiel ROME / e-CF complet.",
        "La similarite multilingue presente un plancher de proximite entre phrases du meme domaine, ce qui demande une calibration du seuil.",
        "L'evaluation de la qualite des textes generes repose sur des metriques simples (couverture, lisibilite).",
    ]))
    story.append(H3("Perspectives"))
    story.extend(bullets([
        "Integrer un referentiel officiel etendu et des profils de metiers plus nombreux.",
        "Ajouter un stockage en base de donnees (SQLite/PostgreSQL) et une authentification.",
        "Affiner le scoring avec un fine-tuning du modele d'embeddings sur des donnees metier.",
        "Enrichir l'evaluation des sorties GenAI (mesures de pertinence et de factualite).",
    ]))

    # ---------- 16. Conclusion ----------
    story.append(H1("16. Conclusion"))
    story.append(P(
        "AISCA demontre un pipeline complet et coherent d'IA generative et de NLP semantique : de la collecte "
        "de competences en langage naturel jusqu'a la recommandation de metiers et la generation guidee de "
        "contenu. Le projet respecte l'ensemble des exigences fonctionnelles (EF1 a EF4), met en oeuvre une "
        "architecture RAG fiable et econome, et fournit une interface professionnelle de demonstration. Il "
        "constitue un prototype representatif des assistants intelligents deployes en HRTech et EdTech, et "
        "valide concretement les competences du Bloc 2 du referentiel RNCP40875."))

    # ---------- Annexes ----------
    story.append(H1("Annexes"))
    story.append(H2("A. Structure du projet"))
    story.append(code(
        ".\n"
        "|-- app.py                  interface Streamlit (5 onglets + sidebar)\n"
        "|-- requirements.txt        dependances\n"
        "|-- .env.example            variables d'environnement\n"
        "|-- data/\n"
        "|   |-- raw/                referentiel brut (CSV) + profils d'exemple (JSON)\n"
        "|   `-- processed/          donnees nettoyees + soumissions\n"
        "|-- cache/                  cache et journal des generations GenAI\n"
        "|-- scripts/                preparation des donnees, demo, generation du rapport\n"
        "|-- src/\n"
        "|   |-- config.py           configuration et chemins\n"
        "|   |-- data_pipeline.py    preparation, preuves, stockage\n"
        "|   |-- semantic_engine.py  embeddings SBERT + cosinus\n"
        "|   |-- recommender.py      scoring et recommandation\n"
        "|   |-- genai.py            agent RAG, cache, fallback\n"
        "|   `-- analytics.py        indicateurs et evaluation\n"
        "`-- tests/                  tests automatises"))
    story.append(H2("B. Stack technique"))
    story.append(P("Python, Streamlit, Pandas, NumPy, Plotly, scikit-learn, SentenceTransformers, Transformers, Google GenAI SDK."))
    story.append(H2("C. Commandes principales"))
    story.append(code(
        "pip install -r requirements.txt        # installation\n"
        "python scripts/prepare_data.py         # preparation des donnees\n"
        "streamlit run app.py                   # lancement de l'application\n"
        "python -m pytest tests/                # execution des tests\n"
        "python scripts/generate_report.py      # generation de ce rapport"))

    return story


def main():
    ref, jobs, kpis, example = load_project_data()

    doc = BaseDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm, bottomMargin=1.8 * cm,
        title="AISCA - Rapport de projet", author=AUTHORS,
    )
    frame_cover = Frame(2 * cm, 2 * cm, A4[0] - 4 * cm, A4[1] - 4 * cm, id="cover")
    frame_main = Frame(2 * cm, 1.8 * cm, A4[0] - 4 * cm, A4[1] - 3.8 * cm, id="main")
    doc.addPageTemplates([
        PageTemplate(id="cover", frames=[frame_cover], onPage=on_cover),
        PageTemplate(id="main", frames=[frame_main], onPage=on_page),
    ])

    doc.build(build_story(ref, jobs, kpis, example))
    print("PDF genere :", OUTPUT)


if __name__ == "__main__":
    main()
