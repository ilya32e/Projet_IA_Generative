from __future__ import annotations

import hashlib
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import GENAI_MODEL_NAME, SBERT_MODEL_NAME
from src.data_pipeline import ensure_submission_identity, load_reference_data, prepare_all_data, save_submission
from src.genai import GenerationSettings, LocalGenAI
from src.recommender import analyse_submission, build_genai_context, job_gap_analysis
from src.semantic_engine import SemanticEngine


st.set_page_config(page_title="AISCA", layout="wide")


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
            line=dict(color="#1f5f8b", width=3),
            fillcolor="rgba(31,95,139,0.20)",
            name="Couverture",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def build_job_figure(job_scores: pd.DataFrame) -> go.Figure:
    top_jobs = job_scores.head(3).sort_values("final_score")
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
    )
    return fig


def analyse_and_store(
    submission: dict,
    reference_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    engine: SemanticEngine,
) -> None:
    submission = ensure_submission_identity(submission)
    save_submission(submission)
    st.session_state["submission"] = submission
    st.session_state["results"] = analyse_submission(submission, reference_df, jobs_df, engine)
    st.session_state.pop("plan_result", None)
    st.session_state.pop("bio_result", None)


def generation_request_key(submission: dict, kind: str, context: dict) -> str:
    payload = {
        "submission_id": submission.get("submission_id", ""),
        "kind": kind,
        "context": context,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


reference_df, jobs_df = get_reference_data()

available_tools = sorted(
    {
        "Python",
        "Pandas",
        "Plotly",
        "Power BI",
        "Streamlit",
        "SentenceTransformers",
        "Transformers",
        "Excel",
        "SQL",
    }
)

st.title("AISCA - Cartographie des competences")
st.write(
    "Application web de questionnaire hybride avec analyse semantique locale, "
    "score de couverture par bloc et recommandation des 3 metiers les plus pertinents."
)

tabs = st.tabs(["Questionnaire", "Resultats", "GenAI"])

with tabs[0]:
    st.subheader("Questionnaire utilisateur")
    st.caption("Le formulaire combine questions ouvertes, niveaux, choix simples et choix multiples.")

    with st.form("questionnaire_form"):
        col1, col2 = st.columns(2)

        with col1:
            candidate_name = st.text_input("Nom du candidat", value="Profil etudiant")
            target_role = st.selectbox("Role vise", options=jobs_df["job_title"].tolist(), index=0)
            education_level = st.text_input("Formation", value="M1 Data / IA")
            experience_months = st.slider("Experience en mois", min_value=0, max_value=36, value=8)
            tools = st.multiselect(
                "Outils utilises",
                options=available_tools,
                default=["Python", "Pandas", "Plotly"],
            )
            project_focus = st.selectbox(
                "Type de projet dominant",
                options=[
                    "Analyse de donnees",
                    "Visualisation / dashboard",
                    "NLP semantique",
                    "IA generative / RAG",
                ],
                index=0,
            )

        with col2:
            focus_blocks = st.multiselect(
                "Blocs a mettre en avant",
                options=reference_df["block_name"].drop_duplicates().tolist(),
                default=["Preparation des donnees", "Analyse exploratoire"],
            )
            tokenization_used = st.radio(
                "Avez-vous deja utilise des techniques de tokenization ?",
                options=["Non", "Notions", "Oui"],
                horizontal=True,
            )
            python_level = st.slider("Niveau Python", 0, 5, 4)
            visualisation_level = st.slider("Niveau visualisation", 0, 5, 4)
            eda_level = st.slider("Niveau analyse exploratoire", 0, 5, 4)
            semantic_nlp_level = st.slider("Niveau NLP semantique", 0, 5, 3)
            genai_level = st.slider("Niveau IA generative / RAG", 0, 5, 3)

        project_text = st.text_area(
            "Decrivez un projet ou vous avez utilise vos competences",
            value="J ai nettoye des donnees, harmonise les colonnes et prepare un jeu de donnees avant analyse.",
            height=120,
        )
        dashboard_text = st.text_area(
            "Decrivez une visualisation ou un dashboard realise",
            value="J ai cree un tableau de bord avec des filtres simples pour presenter les resultats a un manager.",
            height=100,
        )
        genai_text = st.text_area(
            "Decrivez un usage de l IA generative ou du RAG",
            value="J ai utilise un assistant pour produire une synthese et un plan de progression a partir des competences detectees.",
            height=100,
        )

        submitted = st.form_submit_button("Analyser mon profil", use_container_width=True)

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
            engine = get_engine()
            analyse_and_store(submission, reference_df, jobs_df, engine)
            st.session_state["engine_label"] = "SBERT" if engine.info().backend == "sbert" else "Fallback local"
        st.success("Le profil a ete enregistre puis analyse.")

with tabs[1]:
    results = st.session_state.get("results")
    submission = st.session_state.get("submission")

    if not results:
        st.info("Remplissez le questionnaire pour lancer l analyse semantique.")
    else:
        top_job = results["top_jobs"].iloc[0]["job_title"] if not results["top_jobs"].empty else "Aucun"
        engine_label = st.session_state.get("engine_label", "SBERT")
        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Score global", f"{results['final_score']:.2f}")
        metric_2.metric("Top metier recommande", top_job)
        metric_3.metric("Moteur semantique", engine_label)

        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.subheader("Couverture par bloc")
            st.plotly_chart(build_radar_figure(results["block_scores"]), use_container_width=True)
        with chart_right:
            st.subheader("Top 3 metiers recommandes")
            st.plotly_chart(build_job_figure(results["job_scores"]), use_container_width=True)

        st.subheader("Scores par bloc")
        st.dataframe(
            results["block_scores"][["block_name", "block_score", "coverage_rate", "score_label"]],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Classement des metiers")
        st.dataframe(
            results["top_jobs"][["job_title", "final_score", "score_label", "missing_count", "required_count"]],
            use_container_width=True,
            hide_index=True,
        )

        selected_job_title = st.selectbox(
            "Voir les competences a renforcer pour un metier",
            options=results["top_jobs"]["job_title"].tolist(),
        )
        selected_job_id = results["top_jobs"].loc[
            results["top_jobs"]["job_title"] == selected_job_title, "job_id"
        ].iloc[0]
        job_gaps = job_gap_analysis(results["scored_competencies"], jobs_df, selected_job_id)
        st.dataframe(
            job_gaps[["competency_text", "similarity_score", "coverage_label"]],
            use_container_width=True,
            hide_index=True,
        )

        strongest_block = results["block_scores"].iloc[0]
        weakest_block = results["block_scores"].iloc[-1]
        st.write(
            f"Le profil de **{submission['candidate_name']}** est plus solide sur **{strongest_block['block_name']}** "
            f"(score {strongest_block['block_score']:.2f}) et doit surtout progresser sur **{weakest_block['block_name']}** "
            f"(score {weakest_block['block_score']:.2f})."
        )

with tabs[2]:
    results = st.session_state.get("results")
    submission = st.session_state.get("submission")

    if not results:
        st.info("Lancez d abord une analyse pour generer le plan et la bio.")
    else:
        st.subheader("Generation du plan et de la bio")
        st.caption("Cache local actif. Un seul appel pour le plan et un seul appel pour la bio par profil analyse.")

        context = build_genai_context(results, submission)

        col1, col2 = st.columns(2)
        if col1.button("Generer le plan de progression", use_container_width=True):
            generator = get_generator()
            settings = GenerationSettings(max_new_tokens=140, temperature=0.2)
            plan_key = generation_request_key(submission, "plan", context)
            st.session_state["plan_result"] = generator.generate_once("plan", context, plan_key, settings)
        if col2.button("Generer la bio professionnelle", use_container_width=True):
            generator = get_generator()
            settings = GenerationSettings(max_new_tokens=140, temperature=0.2)
            bio_key = generation_request_key(submission, "bio", context)
            st.session_state["bio_result"] = generator.generate_once("bio", context, bio_key, settings)

        if "plan_result" in st.session_state:
            plan_result = st.session_state["plan_result"]
            st.markdown(f"**Plan de progression** - mode `{plan_result['mode']}`")
            st.text_area("Texte genere", value=plan_result["text"], height=220)

        if "bio_result" in st.session_state:
            bio_result = st.session_state["bio_result"]
            st.markdown(f"**Bio professionnelle** - mode `{bio_result['mode']}`")
            st.text_area("Texte genere ", value=bio_result["text"], height=140)
