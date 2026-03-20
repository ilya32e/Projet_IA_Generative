from __future__ import annotations

import json
import re
import hashlib
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError

from .config import (
    BLOCK_NAME_MAP,
    COMPETENCY_PROCESSED_PATH,
    COMPETENCY_RAW_PATH,
    JOB_PROCESSED_PATH,
    JOB_RAW_PATH,
    LEVEL_LABELS,
    SAMPLE_PROFILES_PATH,
    SUBMISSION_INDEX_PATH,
    SUBMISSIONS_DIR,
    ensure_directories,
)


def ascii_slug(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(text))
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", normalized.lower()).strip("_")
    return normalized


def clean_text(text: Any) -> str:
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_competency_reference(
    raw_path: Path = COMPETENCY_RAW_PATH,
    processed_path: Path = COMPETENCY_PROCESSED_PATH,
) -> pd.DataFrame:
    ensure_directories()
    df = pd.read_csv(raw_path)
    df["competency_id"] = df["competency_id"].map(clean_text)
    df["block_id"] = df["block_id"].map(clean_text).str.upper()
    df["block_name"] = df["block_id"].map(BLOCK_NAME_MAP)
    df["competency_text"] = df["competency_text"].map(clean_text)
    df["source"] = df["source"].map(clean_text)
    df["priority"] = df["priority"].map(clean_text).str.lower()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0).clip(0.8, 1.3)
    df["competency_slug"] = df["competency_text"].map(ascii_slug)
    df["word_count"] = df["competency_text"].str.split().str.len()
    df["clean_key"] = df["block_id"] + "__" + df["competency_slug"]
    df = df.drop_duplicates(subset=["clean_key"], keep="first").copy()
    df = df.sort_values(["block_id", "competency_id"]).reset_index(drop=True)
    df = df.drop(columns=["clean_key"])
    df.to_csv(processed_path, index=False)
    return df


def prepare_job_profiles(
    raw_path: Path = JOB_RAW_PATH,
    processed_path: Path = JOB_PROCESSED_PATH,
) -> pd.DataFrame:
    ensure_directories()
    df = pd.read_csv(raw_path)
    df["job_id"] = df["job_id"].map(clean_text)
    df["job_title"] = df["job_title"].map(clean_text)
    df["description"] = df["description"].map(clean_text)
    df["required_competencies"] = df["required_competencies"].map(clean_text)
    df["focus_blocks"] = df["focus_blocks"].map(clean_text)
    df["priority_weight"] = pd.to_numeric(df["priority_weight"], errors="coerce").fillna(1.0).clip(0.8, 1.4)
    df["required_count"] = df["required_competencies"].str.split(";").str.len()
    df["focus_block_count"] = df["focus_blocks"].str.split(";").str.len()
    df.to_csv(processed_path, index=False)
    return df


def prepare_all_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    competencies = prepare_competency_reference()
    jobs = prepare_job_profiles()
    return competencies, jobs


def load_reference_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not COMPETENCY_PROCESSED_PATH.exists() or not JOB_PROCESSED_PATH.exists():
        return prepare_all_data()
    try:
        return pd.read_csv(COMPETENCY_PROCESSED_PATH), pd.read_csv(JOB_PROCESSED_PATH)
    except EmptyDataError:
        return prepare_all_data()


def load_sample_profiles() -> list[dict[str, Any]]:
    with open(SAMPLE_PROFILES_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_submission_identity(submission: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(submission)
    if prepared.get("submission_id") and prepared.get("created_at"):
        return prepared

    identity_payload = {
        "candidate_name": clean_text(prepared.get("candidate_name", "")),
        "target_role": clean_text(prepared.get("target_role", "")),
        "experience_months": int(prepared.get("experience_months", 0)),
        "tools": list(prepared.get("tools", [])),
        "focus_blocks": list(prepared.get("focus_blocks", [])),
        "project_focus": clean_text(prepared.get("project_focus", "")),
        "tokenization_used": clean_text(prepared.get("tokenization_used", "")),
        "levels": prepared.get("levels", {}),
        "project_text": clean_text(prepared.get("project_text", "")),
        "dashboard_text": clean_text(prepared.get("dashboard_text", "")),
        "genai_text": clean_text(prepared.get("genai_text", "")),
        "profile_enrichment": clean_text(prepared.get("profile_enrichment", "")),
    }
    digest = hashlib.sha1(json.dumps(identity_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
    prepared["submission_id"] = prepared.get("submission_id") or f"sub_{digest}"
    prepared["created_at"] = prepared.get("created_at") or datetime.now().isoformat(timespec="seconds")
    return prepared


def build_evidence_texts(submission: dict[str, Any]) -> list[str]:
    levels = submission.get("levels", {})
    tools = submission.get("tools", [])
    focus_blocks = submission.get("focus_blocks", [])
    project_focus = clean_text(submission.get("project_focus", ""))
    tokenization_used = clean_text(submission.get("tokenization_used", ""))
    evidence: list[str] = []

    if tools:
        evidence.append("Outils declares : " + ", ".join(tools))
    if focus_blocks:
        evidence.append("Blocs de competences prioritaires : " + ", ".join(focus_blocks))
    if project_focus:
        evidence.append("Type de projet dominant : " + project_focus)
    if tokenization_used:
        if tokenization_used.lower() == "oui":
            evidence.append("J ai deja utilise des techniques de tokenization dans un projet de traitement de texte.")
        elif tokenization_used.lower() == "non":
            evidence.append("Je n ai pas encore utilise de techniques de tokenization dans un projet.")
        else:
            evidence.append("J ai seulement quelques notions de tokenization et de pretraitement de texte.")

    level_templates = {
        "python": {
            "base": "Niveau en Python : {score}/5 ({label}).",
            "low": "Je connais les bases de Python et pandas pour ouvrir un fichier et faire un premier nettoyage.",
            "mid": "J utilise Python et pandas pour nettoyer des donnees, corriger des doublons et harmoniser des colonnes.",
            "high": "J utilise Python et pandas pour nettoyer, transformer et documenter un jeu de donnees avant analyse.",
        },
        "visualisation": {
            "base": "Niveau en visualisation et dashboard : {score}/5 ({label}).",
            "low": "Je sais produire quelques graphiques simples pour presenter un resultat.",
            "mid": "Je choisis un graphique adapte a la question metier et je construis un tableau de bord clair.",
            "high": "Je construis des tableaux de bord clairs avec filtres interactifs, couleurs lisibles et legende explicite.",
        },
        "eda": {
            "base": "Niveau en analyse exploratoire : {score}/5 ({label}).",
            "low": "Je sais lire quelques indicateurs descriptifs et repere des tendances simples.",
            "mid": "Je calcule des statistiques descriptives et j identifie des blocs forts et faibles a partir des scores.",
            "high": "Je calcule des statistiques descriptives, formule des insights metier et documente les limites de l analyse.",
        },
        "semantic_nlp": {
            "base": "Niveau en NLP semantique : {score}/5 ({label}).",
            "low": "Je connais le principe des embeddings et de la similarite cosinus.",
            "mid": "J encode des phrases et je compare un profil utilisateur a des competences avec une similarite cosinus.",
            "high": "J encode des phrases avec SBERT, je calcule la similarite cosinus et j agrege les scores par bloc de competences.",
        },
        "genai": {
            "base": "Niveau en IA generative et RAG : {score}/5 ({label}).",
            "low": "J ai deja teste un assistant de texte pour reformuler une synthese courte.",
            "mid": "J utilise un assistant pour generer une bio et un plan de progression a partir des competences detectees.",
            "high": "Je construis un contexte RAG, je genere un plan de progression en un appel et j evalue la qualite des sorties.",
        },
    }

    for key, templates in level_templates.items():
        score = int(levels.get(key, 0))
        if score > 0:
            evidence.append(templates["base"].format(score=score, label=LEVEL_LABELS.get(score, "Intermediaire")))
            if score <= 2:
                evidence.append(templates["low"])
            elif score == 3:
                evidence.append(templates["mid"])
            else:
                evidence.append(templates["high"])

    for field in ("project_text", "dashboard_text", "genai_text", "profile_enrichment"):
        value = clean_text(submission.get(field, ""))
        if value:
            evidence.append(value)

    if not evidence:
        evidence.append("Profil debutant sans informations detaillees.")
    return evidence


def flatten_submission(submission: dict[str, Any]) -> dict[str, Any]:
    levels = submission.get("levels", {})
    return {
        "submission_id": submission["submission_id"],
        "candidate_name": clean_text(submission.get("candidate_name", "")),
        "target_role": clean_text(submission.get("target_role", "")),
        "experience_months": int(submission.get("experience_months", 0)),
        "tools": ";".join(submission.get("tools", [])),
        "focus_blocks": ";".join(submission.get("focus_blocks", [])),
        "project_focus": clean_text(submission.get("project_focus", "")),
        "tokenization_used": clean_text(submission.get("tokenization_used", "")),
        "python_level": int(levels.get("python", 0)),
        "visualisation_level": int(levels.get("visualisation", 0)),
        "eda_level": int(levels.get("eda", 0)),
        "semantic_nlp_level": int(levels.get("semantic_nlp", 0)),
        "genai_level": int(levels.get("genai", 0)),
        "project_text_length": len(clean_text(submission.get("project_text", "")).split()),
        "dashboard_text_length": len(clean_text(submission.get("dashboard_text", "")).split()),
        "genai_text_length": len(clean_text(submission.get("genai_text", "")).split()),
        "profile_enrichment_length": len(clean_text(submission.get("profile_enrichment", "")).split()),
        "created_at": submission["created_at"],
    }


def save_submission(submission: dict[str, Any]) -> Path:
    ensure_directories()
    prepared = ensure_submission_identity(submission)
    output_path = SUBMISSIONS_DIR / f"{prepared['submission_id']}.json"

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(prepared, handle, indent=2, ensure_ascii=False)

    row = pd.DataFrame([flatten_submission(prepared)])
    if SUBMISSION_INDEX_PATH.exists():
        current = pd.read_csv(SUBMISSION_INDEX_PATH)
        current = pd.concat([current, row], ignore_index=True)
        current = current.drop_duplicates(subset=["submission_id"], keep="last")
    else:
        current = row
    current.to_csv(SUBMISSION_INDEX_PATH, index=False)
    return output_path


def load_saved_submissions() -> list[dict[str, Any]]:
    ensure_directories()
    submissions: list[dict[str, Any]] = []
    for path in sorted(SUBMISSIONS_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            submissions.append(json.load(handle))
    return submissions
