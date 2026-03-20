from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import SEMANTIC_THRESHOLD, TOP_N_JOBS
from .data_pipeline import build_evidence_texts, clean_text


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(np.average(values, weights=weights))


def _score_label(score: float) -> str:
    if score >= 0.75:
        return "Fort"
    if score >= 0.55:
        return "Correct"
    if score >= 0.35:
        return "Moyen"
    return "A renforcer"


def score_competencies(
    submission: dict[str, Any],
    reference_df: pd.DataFrame,
    engine: Any,
) -> tuple[pd.DataFrame, list[str]]:
    evidence_texts = build_evidence_texts(submission)
    similarity_matrix = engine.pairwise_similarity(evidence_texts, reference_df["competency_text"].tolist())

    if similarity_matrix.size == 0:
        scored = reference_df.copy()
        scored["similarity_score"] = 0.0
        scored["best_evidence"] = ""
        scored["coverage_label"] = "A renforcer"
        return scored, evidence_texts

    best_indices = similarity_matrix.argmax(axis=0)
    raw_scores = similarity_matrix.max(axis=0)
    best_scores = np.sqrt(np.clip(raw_scores, 0.0, 1.0))

    scored = reference_df.copy()
    scored["raw_similarity"] = np.round(raw_scores, 4)
    scored["similarity_score"] = np.round(best_scores, 4)
    scored["best_evidence"] = [evidence_texts[index] for index in best_indices]
    scored["coverage_label"] = scored["similarity_score"].map(_score_label)
    return scored.sort_values("similarity_score", ascending=False).reset_index(drop=True), evidence_texts


def aggregate_block_scores(scored_df: pd.DataFrame, threshold: float = SEMANTIC_THRESHOLD) -> pd.DataFrame:
    block_rows = []
    for _, block_df in scored_df.groupby("block_id"):
        score = _weighted_average(block_df["similarity_score"], block_df["weight"])
        coverage_rate = float((block_df["similarity_score"] >= threshold).mean())
        block_rows.append(
            {
                "block_id": block_df["block_id"].iloc[0],
                "block_name": block_df["block_name"].iloc[0],
                "block_score": round(score, 4),
                "coverage_rate": round(coverage_rate, 4),
                "skills_count": int(len(block_df)),
                "strong_skills": int((block_df["similarity_score"] >= threshold).sum()),
            }
        )
    block_scores = pd.DataFrame(block_rows).sort_values("block_score", ascending=False).reset_index(drop=True)
    block_scores["score_label"] = block_scores["block_score"].map(_score_label)
    return block_scores


def compute_job_scores(
    scored_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    threshold: float = SEMANTIC_THRESHOLD,
) -> pd.DataFrame:
    rows = []
    indexed_scores = scored_df.set_index("competency_id")
    block_scores = scored_df.groupby("block_id")["similarity_score"].mean()

    for _, job in jobs_df.iterrows():
        competency_ids = [clean_text(item) for item in str(job["required_competencies"]).split(";") if clean_text(item)]
        job_slice = indexed_scores.loc[indexed_scores.index.intersection(competency_ids)]
        if job_slice.empty:
            average_score = 0.0
            coverage_rate = 0.0
            missing_count = len(competency_ids)
        else:
            average_score = float(job_slice["similarity_score"].mean())
            coverage_rate = float((job_slice["similarity_score"] >= threshold).mean())
            missing_count = int((job_slice["similarity_score"] < threshold).sum())

        focus_blocks = [clean_text(item).upper() for item in str(job["focus_blocks"]).split(";") if clean_text(item)]
        block_bonus = float(block_scores.reindex(focus_blocks).fillna(0).mean()) if focus_blocks else 0.0
        final_score = (0.55 * average_score) + (0.25 * coverage_rate) + (0.20 * block_bonus)
        final_score *= float(job["priority_weight"])

        rows.append(
            {
                "job_id": job["job_id"],
                "job_title": job["job_title"],
                "description": job["description"],
                "average_score": round(average_score, 4),
                "coverage_rate": round(coverage_rate, 4),
                "block_bonus": round(block_bonus, 4),
                "final_score": round(min(final_score, 1.0), 4),
                "missing_count": missing_count,
                "required_count": int(job["required_count"]),
            }
        )

    ranking = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    ranking["score_label"] = ranking["final_score"].map(_score_label)
    return ranking


def job_gap_analysis(scored_df: pd.DataFrame, jobs_df: pd.DataFrame, job_id: str, limit: int = 5) -> pd.DataFrame:
    indexed = scored_df.set_index("competency_id")
    job_row = jobs_df.loc[jobs_df["job_id"] == job_id].iloc[0]
    competency_ids = [clean_text(item) for item in str(job_row["required_competencies"]).split(";") if clean_text(item)]
    job_slice = indexed.loc[indexed.index.intersection(competency_ids)].reset_index()
    return job_slice.sort_values("similarity_score", ascending=True).head(limit).reset_index(drop=True)


def build_genai_context(results: dict[str, Any], submission: dict[str, Any]) -> dict[str, Any]:
    top_jobs = results["job_scores"].head(3)[["job_title", "final_score"]].to_dict(orient="records")
    block_scores = results["block_scores"][["block_name", "block_score"]].to_dict(orient="records")
    strengths = results["scored_competencies"].head(4)["competency_text"].tolist()
    gaps = results["target_gaps"].head(4)["competency_text"].tolist()
    return {
        "candidate_name": submission.get("candidate_name", "Profil etudiant"),
        "target_role": submission.get("target_role", "Role non precise"),
        "top_jobs": top_jobs,
        "block_scores": block_scores,
        "strengths": strengths,
        "gaps": gaps,
        "evidence": results["evidence_texts"][:4],
        "overall_score": round(results["final_score"], 4),
    }


def analyse_submission(
    submission: dict[str, Any],
    reference_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    engine: Any,
    threshold: float = SEMANTIC_THRESHOLD,
) -> dict[str, Any]:
    scored_competencies, evidence_texts = score_competencies(submission, reference_df, engine)
    block_scores = aggregate_block_scores(scored_competencies, threshold=threshold)
    job_scores = compute_job_scores(
        scored_competencies,
        jobs_df,
        threshold=threshold,
    )
    top_jobs = job_scores.head(TOP_N_JOBS).copy()
    top_job_id = top_jobs.iloc[0]["job_id"] if not top_jobs.empty else jobs_df.iloc[0]["job_id"]
    target_gaps = job_gap_analysis(scored_competencies, jobs_df, top_job_id)
    final_score = float(np.average(block_scores["block_score"], weights=block_scores["coverage_rate"] + 0.2))

    return {
        "submission": submission,
        "evidence_texts": evidence_texts,
        "scored_competencies": scored_competencies,
        "block_scores": block_scores,
        "job_scores": job_scores,
        "top_jobs": top_jobs,
        "top_job_id": top_job_id,
        "target_gaps": target_gaps,
        "final_score": round(min(final_score, 1.0), 4),
    }
