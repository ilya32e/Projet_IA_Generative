from __future__ import annotations

import hashlib
import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics import (
    build_sample_heatmap,
    evaluate_generated_text,
    load_generation_history,
    reference_kpis,
    submission_overview,
)
from src.config import GENAI_MODEL_NAME, SBERT_MODEL_NAME, SEMANTIC_THRESHOLD, TOP_N_JOBS
from src.data_pipeline import (
    ensure_submission_identity,
    load_reference_data,
    load_sample_profiles,
    load_saved_submissions,
    prepare_all_data,
    save_submission,
)
from src.genai import GenerationSettings, LocalGenAI
from src.recommender import analyse_submission, build_genai_context, job_gap_analysis
from src.semantic_engine import SemanticEngine

st.set_page_config(page_title="AISCA", page_icon="🧭", layout="wide")

PRIMARY = "#12395a"
ACCENT = "#1f7a8c"


# --------------------------------------------------------------------------- #
# Ressources mises en cache
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_reference_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    prepare_all_data()
    return load_reference_data()


@st.cache_resource(show_spinner=False)
def get_engine() -> SemanticEngine:
    return SemanticEngine(backend="auto", model_name=SBERT_MODEL_NAME)


@st.cache_resource(show_spinner=False)
def get_generator() -> LocalGenAI:
    return LocalGenAI(model_name=GENAI_MODEL_NAME)


@st.cache_data(show_spinner=False)
def get_sample_profiles() -> list[dict]:
    return load_sample_profiles()


# --------------------------------------------------------------------------- #
# Visualisations
# --------------------------------------------------------------------------- #
def build_radar_figure(block_scores: pd.DataFrame) -> go.Figure:
    labels = block_scores["block_name"].tolist()
    values = block_scores["block_score"].tolist()
    if labels:
        labels = labels + [labels[0]]
        values = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill="toself",
            line=dict(color=ACCENT, width=3),
            fillcolor="rgba(31,122,140,0.20)",
            name="Couverture",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=360,
    )
    return fig


def build_job_figure(job_scores: pd.DataFrame, top_n: int = 3) -> go.Figure:
    top_jobs = job_scores.head(top_n).sort_values("final_score")
    fig = px.bar(
        top_jobs,
        x="final_score",
        y="job_title",
        orientation="h",
        text="final_score",
        color="score_label",
        color_discrete_map={
            "Fort": "#1f7a8c",
            "Correct": "#f4a261",
            "Moyen": "#c97b63",
            "A renforcer": "#c0392b",
        },
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(
        xaxis_title="Score de couverture",
        yaxis_title="Metiers",
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        height=360,
    )
    return fig


def build_heatmap_figure(heatmap_df: pd.DataFrame) -> go.Figure:
    pivot = heatmap_df.pivot(index="candidate_name", columns="block_name", values="block_score")
    fig = px.imshow(
        pivot,
        text_auto=".2f",
        color_continuous_scale="Tealgrn",
        aspect="auto",
        zmin=0,
        zmax=1,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=360,
        coloraxis_colorbar=dict(title="Score"),
        xaxis_title="",
        yaxis_title="",
    )
    return fig


# --------------------------------------------------------------------------- #
# Helpers metier
# --------------------------------------------------------------------------- #
def status_dot(level: str) -> str:
    return {"ok": "🟢", "warn": "🟡", "off": "🔴"}.get(level, "⚪")


def generation_request_key(submission: dict, kind: str, context: dict) -> str:
    payload = {"submission_id": submission.get("submission_id", ""), "kind": kind, "context": context}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def analyse_and_store(submission, reference_df, jobs_df, engine, enrich=False, generator=None):
    if enrich and generator is not None:
        submission = generator.enrich_submission_if_needed(submission)
    submission = ensure_submission_identity(submission)
    save_submission(submission)
    st.session_state["submission"] = submission
    st.session_state["results"] = analyse_submission(submission, reference_df, jobs_df, engine)
    st.session_state["engine_label"] = "SBERT" if engine.info().backend == "sbert" else "Fallback lexical"
    st.session_state.pop("plan_result", None)
    st.session_state.pop("bio_result", None)


def build_markdown_report(results, submission, plan_text, bio_text) -> str:
    lines = [
        f"# Rapport AISCA — {submission.get('candidate_name', 'Profil')}",
        "",
        f"- Genere le : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"- Role vise : {submission.get('target_role', 'Non precise')}",
        f"- Formation : {submission.get('education_level', '')}",
        f"- Score global de couverture semantique : **{results['final_score']:.2f}**",
        "",
        "## Top metiers recommandes",
    ]
    for i, row in enumerate(results["top_jobs"].itertuples(), start=1):
        lines.append(f"{i}. {row.job_title} — score {row.final_score:.2f} ({row.score_label})")
    lines += ["", "## Scores par bloc de competences"]
    for row in results["block_scores"].itertuples():
        lines.append(f"- {row.block_name} : {row.block_score:.2f} ({row.score_label})")
    lines += ["", "## Competences prioritaires a renforcer"]
    for row in results["target_gaps"].itertuples():
        lines.append(f"- {row.competency_text} (similarite {row.similarity_score:.2f})")
    if plan_text:
        lines += ["", "## Plan de progression (GenAI)", "", plan_text]
    if bio_text:
        lines += ["", "## Bio professionnelle (GenAI)", "", bio_text]
    lines += ["", "---", "_Genere par AISCA — agent semantique et generatif (mini-RAG, SBERT local + GenAI controlee)._"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Chargement
# --------------------------------------------------------------------------- #
reference_df, jobs_df = get_reference_data()
kpis = reference_kpis(reference_df, jobs_df)

available_tools = sorted(
    {
        # Langages & manipulation
        "Python", "R", "SQL", "Excel", "Pandas", "NumPy",
        # Machine learning
        "scikit-learn", "PyTorch", "TensorFlow", "Keras", "XGBoost",
        # NLP & IA generative
        "Transformers", "SentenceTransformers", "spaCy", "NLTK", "Gensim",
        "Hugging Face", "LangChain", "LlamaIndex", "FAISS", "ChromaDB",
        "OpenAI API", "Gemini API", "Ollama",
        # Visualisation & BI
        "Matplotlib", "Seaborn", "Plotly", "Power BI", "Tableau", "Looker",
        # Web & applications
        "Streamlit", "Flask", "FastAPI", "Django", "Gradio",
        # Data engineering & outils
        "Git", "Docker", "Jupyter", "VS Code", "Airflow", "Spark", "MongoDB", "PostgreSQL",
    }
)


# --------------------------------------------------------------------------- #
# Barre laterale : etat du systeme
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🧭 AISCA")
    st.caption("Agent Intelligent Semantique et Generatif — Cartographie des competences")

    engine = get_engine()
    engine_info = engine.info()
    generator = get_generator()
    gen_status = generator.status()

    st.markdown("#### Etat du systeme")
    sbert_ok = "ok" if engine_info.backend == "sbert" else "warn"
    st.markdown(
        f"{status_dot(sbert_ok)} **Moteur semantique** — "
        f"{'SBERT local' if engine_info.backend == 'sbert' else 'Fallback lexical (TF-IDF)'}"
    )
    st.caption(f"Modele : `{engine_info.model_name}`")

    if gen_status["gemini_ready"]:
        gen_dot, gen_lbl = "ok", f"Gemini actif — `{gen_status['model_name']}`"
    else:
        gen_dot, gen_lbl = "warn", "GenAI en repli local / template"
    st.markdown(f"{status_dot(gen_dot)} **GenAI** — {gen_lbl}")
    st.caption(f"Cache GenAI : {gen_status['cache_entries']} entree(s)")

    st.markdown("#### Referentiel")
    c1, c2 = st.columns(2)
    c1.metric("Competences", kpis["competencies"])
    c2.metric("Blocs", kpis["blocks"])
    c1.metric("Metiers", kpis["jobs"])
    c2.metric("Seuil", f"{SEMANTIC_THRESHOLD:.2f}")

    st.divider()
    if st.button("🗑️ Vider le cache GenAI", use_container_width=True):
        from src.config import GENAI_CACHE_PATH

        if GENAI_CACHE_PATH.exists():
            GENAI_CACHE_PATH.unlink()
        get_generator.clear()
        st.session_state.pop("plan_result", None)
        st.session_state.pop("bio_result", None)
        st.rerun()


# --------------------------------------------------------------------------- #
# En-tete
# --------------------------------------------------------------------------- #
st.title("AISCA — Cartographie semantique des competences")
st.caption(
    "Mini-agent RAG : appariement semantique SBERT → score de couverture pondere → "
    "recommandation des 3 metiers → generation controlee (plan + bio). "
    "RNCP40875 · Bloc 2 — Piloter et implementer des solutions d'IA."
)

tabs = st.tabs(["📋 Questionnaire", "📊 Resultats", "🤖 Agent RAG", "📈 Analytics", "ℹ️ Methodologie"])


# --------------------------------------------------------------------------- #
# Onglet 1 — Questionnaire
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.subheader("Questionnaire hybride")
    st.caption("Texte libre (preuves contextuelles), auto-evaluation Likert, question guidee et choix multiples — EF1.1.")

    sample_profiles = get_sample_profiles()
    preset_names = ["— Saisie manuelle —"] + [p["candidate_name"] for p in sample_profiles]
    preset_choice = st.selectbox("Charger un profil d'exemple (demo rapide)", options=preset_names)
    preset = next((p for p in sample_profiles if p["candidate_name"] == preset_choice), None)

    def pv(field, default):
        return preset.get(field, default) if preset else default

    with st.form("questionnaire_form"):
        col1, col2 = st.columns(2)

        with col1:
            candidate_name = st.text_input("Nom du candidat", value=pv("candidate_name", "Profil etudiant"))
            target_role = st.selectbox("Role vise", options=jobs_df["job_title"].tolist())
            education_level = st.selectbox(
                "Formation",
                options=[
                    "Licence Informatique", "Licence Mathematiques / Statistiques", "Licence MIASHS / Data",
                    "BUT Science des donnees", "M1 Data / IA", "M2 Data Science", "M2 Data Engineering",
                    "M2 NLP / IA", "Master MIAGE", "Ecole d ingenieur", "Bootcamp Data / IA",
                    "Autodidacte", "Doctorat",
                ],
                index=4,
            )
            experience_months = st.slider("Experience en mois", 0, 36, int(pv("experience_months", 8)))
            tools = st.multiselect("Outils utilises", options=available_tools, default=pv("tools", ["Python", "Pandas", "Plotly"]))
            project_focus = st.selectbox(
                "Type de projet dominant",
                options=[
                    "Analyse de donnees", "Nettoyage et qualite des donnees", "Data engineering / pipeline",
                    "Statistiques et analyse exploratoire", "Visualisation / dashboard",
                    "Business intelligence / reporting", "Machine learning / modelisation", "NLP semantique",
                    "Recommandation / matching", "IA generative / RAG", "Chatbot / assistant conversationnel",
                    "Evaluation de modeles / metriques", "Cartographie de competences",
                ],
            )

        with col2:
            default_blocks = [b for b in pv("focus_blocks", ["Preparation des donnees", "Analyse exploratoire"]) if b in reference_df["block_name"].unique()]
            focus_blocks = st.multiselect(
                "Blocs a mettre en avant",
                options=reference_df["block_name"].drop_duplicates().tolist(),
                default=default_blocks or ["Preparation des donnees"],
            )
            tokenization_used = st.radio(
                "Avez-vous deja utilise des techniques de tokenization ?",
                options=["Jamais", "Non", "Notions", "Oui", "Oui en projet", "Oui en production"],
                index=2,
                horizontal=True,
            )
            lvl = pv("levels", {})
            python_level = st.slider("Niveau Python", 0, 5, int(lvl.get("python", 4)))
            visualisation_level = st.slider("Niveau visualisation", 0, 5, int(lvl.get("visualisation", 4)))
            eda_level = st.slider("Niveau analyse exploratoire", 0, 5, int(lvl.get("eda", 4)))
            semantic_nlp_level = st.slider("Niveau NLP semantique", 0, 5, int(lvl.get("semantic_nlp", 3)))
            genai_level = st.slider("Niveau IA generative / RAG", 0, 5, int(lvl.get("genai", 3)))

        project_text = st.text_area(
            "Decrivez un projet ou vous avez utilise vos competences",
            value=pv("project_text", "J ai nettoye des donnees, harmonise les colonnes et prepare un jeu de donnees avant analyse."),
            height=110,
        )
        dashboard_text = st.text_area(
            "Decrivez une visualisation ou un dashboard realise",
            value=pv("dashboard_text", "J ai cree un tableau de bord avec des filtres simples pour presenter les resultats a un manager."),
            height=90,
        )
        genai_text = st.text_area(
            "Decrivez un usage de l IA generative ou du RAG",
            value=pv("genai_text", "J ai utilise un assistant pour produire une synthese a partir des competences detectees."),
            height=90,
        )

        enrich_short = st.checkbox(
            "Enrichir automatiquement les reponses trop courtes via GenAI (EF4.1 — appel conditionnel)",
            value=False,
        )
        submitted = st.form_submit_button("🚀 Analyser mon profil", use_container_width=True)

    if submitted:
        submission = {
            "candidate_name": candidate_name,
            "target_role": target_role,
            "education_level": education_level,
            "experience_months": experience_months,
            "tools": tools,
            "focus_blocks": focus_blocks,
            "project_focus": project_focus,
            "tokenization_used": tokenization_used,
            "levels": {
                "python": python_level,
                "visualisation": visualisation_level,
                "eda": eda_level,
                "semantic_nlp": semantic_nlp_level,
                "genai": genai_level,
            },
            "project_text": project_text,
            "dashboard_text": dashboard_text,
            "genai_text": genai_text,
        }
        with st.spinner("Analyse semantique en cours..."):
            analyse_and_store(submission, reference_df, jobs_df, engine, enrich=enrich_short, generator=generator)
        st.success("Profil enregistre puis analyse. Consultez l'onglet **Resultats**.")
        if st.session_state["submission"].get("profile_enrichment"):
            st.info("✨ Reponses courtes enrichies par la GenAI avant l'analyse (EF4.1).")


# --------------------------------------------------------------------------- #
# Onglet 2 — Resultats
# --------------------------------------------------------------------------- #
with tabs[1]:
    results = st.session_state.get("results")
    submission = st.session_state.get("submission")

    if not results:
        st.info("Remplissez le questionnaire pour lancer l'analyse semantique.")
    else:
        top_job = results["top_jobs"].iloc[0]["job_title"] if not results["top_jobs"].empty else "Aucun"
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score global", f"{results['final_score']:.2f}")
        m2.metric("Top metier", top_job)
        m3.metric("Moteur", st.session_state.get("engine_label", "SBERT"))
        m4.metric("Metiers compares", len(results["job_scores"]))

        left, right = st.columns(2)
        with left:
            st.subheader("Couverture par bloc")
            st.plotly_chart(build_radar_figure(results["block_scores"]), use_container_width=True)
        with right:
            st.subheader(f"Top {TOP_N_JOBS} metiers recommandes")
            st.plotly_chart(build_job_figure(results["job_scores"], TOP_N_JOBS), use_container_width=True)

        st.subheader("Scores par bloc")
        st.dataframe(
            results["block_scores"][["block_name", "block_score", "coverage_rate", "score_label"]],
            use_container_width=True, hide_index=True,
        )

        st.subheader("Classement complet des metiers")
        st.dataframe(
            results["job_scores"][["job_title", "average_score", "coverage_rate", "final_score", "score_label", "missing_count"]],
            use_container_width=True, hide_index=True,
        )

        selected_job_title = st.selectbox(
            "Competences a renforcer pour un metier",
            options=results["job_scores"]["job_title"].tolist(),
        )
        selected_job_id = jobs_df.loc[jobs_df["job_title"] == selected_job_title, "job_id"].iloc[0]
        job_gaps = job_gap_analysis(results["scored_competencies"], jobs_df, selected_job_id)
        st.dataframe(
            job_gaps[["competency_text", "similarity_score", "coverage_label"]],
            use_container_width=True, hide_index=True,
        )

        strongest = results["block_scores"].iloc[0]
        weakest = results["block_scores"].iloc[-1]
        st.markdown(
            f"Le profil de **{submission['candidate_name']}** est le plus solide sur "
            f"**{strongest['block_name']}** ({strongest['block_score']:.2f}) et doit surtout progresser sur "
            f"**{weakest['block_name']}** ({weakest['block_score']:.2f})."
        )

        report_md = build_markdown_report(
            results, submission,
            st.session_state.get("plan_result", {}).get("text", ""),
            st.session_state.get("bio_result", {}).get("text", ""),
        )
        st.download_button(
            "⬇️ Telecharger le rapport (Markdown)",
            data=report_md,
            file_name=f"rapport_aisca_{submission.get('submission_id', 'profil')}.md",
            mime="text/markdown",
            use_container_width=True,
        )


# --------------------------------------------------------------------------- #
# Onglet 3 — Agent RAG (Retrieval -> Augmented -> Generation)
# --------------------------------------------------------------------------- #
with tabs[2]:
    results = st.session_state.get("results")
    submission = st.session_state.get("submission")

    if not results:
        st.info("Lancez d'abord une analyse pour activer l'agent RAG.")
    else:
        context = build_genai_context(results, submission)
        st.subheader("Architecture de l'agent RAG")
        st.caption("Chaque generation suit les 3 etapes du RAG, garantissant des sorties fiables et controlees.")

        step1, step2, step3 = st.columns(3)
        with step1:
            st.markdown("#### 1 · Retrieval")
            st.caption("Recuperer uniquement les elements pertinents du referentiel.")
            st.markdown("**Points forts detectes**")
            for s in context["strengths"]:
                st.markdown(f"- {s}")
            st.markdown("**Competences faibles (gaps)**")
            for g in context["gaps"]:
                st.markdown(f"- {g}")
        with step2:
            st.markdown("#### 2 · Augmented context")
            st.caption("Construire un contexte structure transmis au LLM.")
            st.json(
                {
                    "candidate_name": context["candidate_name"],
                    "target_role": context["target_role"],
                    "overall_score": context["overall_score"],
                    "top_jobs": context["top_jobs"],
                    "block_scores": context["block_scores"],
                },
                expanded=False,
            )
        with step3:
            st.markdown("#### 3 · Generation")
            st.caption("Le LLM produit le contenu final a partir du contexte.")
            st.markdown(
                f"Provider : **{'Gemini' if get_generator().status()['gemini_ready'] else 'Repli local/template'}**\n\n"
                "Contraintes : **1 seul appel** par plan et par bio, **cache** automatique."
            )
            if submission.get("profile_enrichment"):
                with st.expander("Pre-processing GenAI (EF4.1) applique"):
                    st.write(submission["profile_enrichment"])

        st.divider()
        st.subheader("Generation controlee")

        gen_left, gen_right = st.columns(2)
        if gen_left.button("📝 Generer le plan de progression", use_container_width=True):
            with st.spinner("Generation du plan..."):
                settings = GenerationSettings(max_new_tokens=180, temperature=0.3)
                key = generation_request_key(submission, "plan", context)
                st.session_state["plan_result"] = get_generator().generate_once("plan", context, key, settings)
        if gen_right.button("👤 Generer la bio professionnelle", use_container_width=True):
            with st.spinner("Generation de la bio..."):
                settings = GenerationSettings(max_new_tokens=140, temperature=0.3)
                key = generation_request_key(submission, "bio", context)
                st.session_state["bio_result"] = get_generator().generate_once("bio", context, key, settings)

        quality_terms = context["strengths"] + context["gaps"] + [j["job_title"] for j in context["top_jobs"]]

        for label, state_key, bounds in [
            ("Plan de progression", "plan_result", (60, 180)),
            ("Bio professionnelle", "bio_result", (50, 110)),
        ]:
            if state_key in st.session_state:
                res = st.session_state[state_key]
                badge = {"gemini_api": "🟢 Gemini", "local_transformers": "🟡 Modele local", "template_fallback": "⚪ Template"}.get(res["mode"], res["mode"])
                cache_badge = "♻️ cache" if res.get("cache_hit") else "🆕 nouvel appel"
                st.markdown(f"**{label}** — {badge} · {cache_badge}")
                st.text_area(f"Texte — {label}", value=res["text"], height=180, key=f"ta_{state_key}")
                metrics = evaluate_generated_text(res["text"], quality_terms, bounds[0], bounds[1])
                q1, q2, q3, q4 = st.columns(4)
                q1.metric("Couverture contexte", f"{metrics['coverage']:.2f}")
                q2.metric("Longueur", f"{int(metrics['word_count'])} mots")
                q3.metric("Lisibilite", f"{metrics['readability']:.2f}")
                q4.metric("Qualite globale", f"{metrics['overall']:.2f}")


# --------------------------------------------------------------------------- #
# Onglet 4 — Analytics
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.subheader("Cartographie des profils d'exemple")
    st.caption("Scores de couverture par bloc pour les profils de reference — utile pour calibrer et demontrer le moteur.")

    with st.spinner("Calcul des scores sur les profils d'exemple..."):
        heatmap_df = build_sample_heatmap(get_sample_profiles(), reference_df, jobs_df, engine)
    st.plotly_chart(build_heatmap_figure(heatmap_df), use_container_width=True)

    top_by_profile = (
        heatmap_df[["candidate_name", "top_job"]].drop_duplicates().rename(columns={"candidate_name": "Profil", "top_job": "Metier recommande"})
    )
    st.dataframe(top_by_profile, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Journal des generations GenAI")
    st.caption("Tracabilite des appels : mode, cache, longueur — preuve du controle des couts (Free Tier).")
    history = load_generation_history()
    if history.empty:
        st.info("Aucune generation enregistree pour le moment. Generez un plan ou une bio dans l'onglet Agent RAG.")
    else:
        h1, h2, h3 = st.columns(3)
        h1.metric("Appels totaux", len(history))
        if "cache_hit" in history:
            hits = pd.to_numeric(history["cache_hit"], errors="coerce").fillna(0).astype(int).sum()
            h2.metric("Hits cache", int(hits))
            h3.metric("Taux de reutilisation", f"{(hits / len(history) * 100):.0f}%")
        st.dataframe(history.tail(20), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Soumissions enregistrees")
    saved = load_saved_submissions()
    overview = submission_overview(saved)
    if overview.empty:
        st.info("Aucune soumission enregistree.")
    else:
        st.dataframe(overview, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Onglet 5 — Methodologie
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.subheader("Methodologie & conformite")
    st.markdown(
        """
**Pipeline complet (NLP → Scoring → Recommandation → GenAI).**

1. **Acquisition (EF1)** — questionnaire hybride (texte libre + Likert + question guidee + choix multiples).
   Les reponses sont normalisees et stockees en JSON/CSV.
2. **Moteur semantique (EF2)** — chaque preuve utilisateur et chaque competence du referentiel sont encodees
   avec **SBERT multilingue** ; on calcule la **similarite cosinus** (repli TF-IDF local si SBERT indisponible).
3. **Scoring (EF3)** — agregation ponderee par bloc, puis score par metier
   `final = 0.60 · similarite_moyenne + 0.40 · taux_de_couverture`, et **top 3** des metiers.
4. **GenAI controlee (EF4)** — agent **RAG** : Retrieval (forces/gaps) → Augmented context → Generation
   du plan et de la bio, **1 appel** chacun, **cache** local, enrichissement conditionnel des reponses courtes.
        """
    )

    st.markdown("#### Couverture des exigences fonctionnelles")
    st.dataframe(
        pd.DataFrame(
            [
                ["EF1.1", "Questionnaire hybride", "Onglet Questionnaire (texte libre, Likert, guidee, multiselect)"],
                ["EF1.2", "Stockage structure", "JSON par soumission + index CSV"],
                ["EF2.1", "Referentiel competences/metiers", f"{kpis['competencies']} competences · {kpis['blocks']} blocs · {kpis['jobs']} metiers (inspire ROME/eCF)"],
                ["EF2.2", "Embeddings SBERT locaux", f"`{engine_info.model_name}`"],
                ["EF2.3", "Similarite cosinus", "semantic_engine.pairwise_similarity"],
                ["EF3.1", "Formule de score ponderee", "0.60·moyenne + 0.40·couverture, seuil " + f"{SEMANTIC_THRESHOLD:.2f}"],
                ["EF3.2", "Top 3 metiers", "Onglet Resultats + graphe barres"],
                ["EF4.1", "Enrichissement conditionnel", "enrich_submission_if_needed (reponses < 18 mots)"],
                ["EF4.2", "Plan de progression", "1 appel verrouille + cache (Agent RAG)"],
                ["EF4.3", "Bio professionnelle", "1 appel verrouille + cache (Agent RAG)"],
            ],
            columns=["Exigence", "Intitule", "Implementation"],
        ),
        use_container_width=True, hide_index=True,
    )

    st.markdown("#### Contraintes GenAI (Free Tier)")
    st.markdown(
        "- **Appels limites** : 1 plan + 1 bio par profil analyse (verrou logique `generate_once`).\n"
        "- **Caching automatique** : `cache/genai_cache.json` ; requetes identiques reutilisees.\n"
        "- **Repli robuste** : Gemini → modele local → template, l'app reste fonctionnelle sans cle API.\n"
        "- **Tracabilite** : journal des appels dans `cache/genai_log.csv` (onglet Analytics)."
    )
