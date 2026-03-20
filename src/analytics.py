from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .config import GENAI_LOG_PATH
from .recommender import analyse_submission


def reference_kpis(reference_df: pd.DataFrame, jobs_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "competencies": int(len(reference_df)),
        "blocks": int(reference_df["block_name"].nunique()),
        "jobs": int(len(jobs_df)),
        "avg_weight": round(float(reference_df["weight"].mean()), 2),
    }


def submission_overview(saved_submissions: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for submission in saved_submissions:
        rows.append(
            {
                "candidate_name": submission.get("candidate_name", "Profil"),
                "experience_months": int(submission.get("experience_months", 0)),
                "tools_count": len(submission.get("tools", [])),
                "focus_block_count": len(submission.get("focus_blocks", [])),
                "project_text_length": len(str(submission.get("project_text", "")).split()),
                "dashboard_text_length": len(str(submission.get("dashboard_text", "")).split()),
                "genai_text_length": len(str(submission.get("genai_text", "")).split()),
            }
        )
    return pd.DataFrame(rows)


def build_sample_heatmap(
    sample_profiles: list[dict[str, Any]],
    reference_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    engine: Any,
) -> pd.DataFrame:
    rows = []
    for profile in sample_profiles:
        results = analyse_submission(profile, reference_df, jobs_df, engine)
        top_job = results["top_jobs"].iloc[0]["job_title"] if not results["top_jobs"].empty else "Aucun"
        for _, block_row in results["block_scores"].iterrows():
            rows.append(
                {
                    "candidate_name": profile["candidate_name"],
                    "block_name": block_row["block_name"],
                    "block_score": block_row["block_score"],
                    "top_job": top_job,
                }
            )
    return pd.DataFrame(rows)


def evaluate_generated_text(
    text: str,
    context_terms: list[str],
    min_words: int,
    max_words: int,
) -> dict[str, float]:
    text = text.strip()
    lowered = text.lower()
    words = re.findall(r"\b\w+\b", lowered)
    sentences = [item.strip() for item in re.split(r"[.!?]+", text) if item.strip()]
    context_terms = [item.lower() for item in context_terms if item]

    coverage_hits = sum(1 for term in context_terms if term.lower() in lowered)
    coverage = coverage_hits / max(len(context_terms), 1)

    word_count = len(words)
    if min_words <= word_count <= max_words:
        length_score = 1.0
    else:
        distance = min(abs(word_count - min_words), abs(word_count - max_words))
        length_score = max(0.0, 1 - (distance / max(max_words, 1)))

    avg_sentence_length = word_count / max(len(sentences), 1)
    readability = max(0.0, 1 - (abs(avg_sentence_length - 18) / 22))
    diversity = len(set(words)) / max(word_count, 1)
    overall = (0.45 * coverage) + (0.25 * length_score) + (0.15 * readability) + (0.15 * diversity)

    return {
        "coverage": round(coverage, 4),
        "length_score": round(length_score, 4),
        "readability": round(readability, 4),
        "diversity": round(diversity, 4),
        "overall": round(overall, 4),
        "word_count": float(word_count),
    }


def load_generation_history(log_path: Path = GENAI_LOG_PATH) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(
            columns=["created_at", "kind", "mode", "temperature", "max_new_tokens", "word_count", "cache_hit"]
        )
    with open(log_path, "r", encoding="utf-8", newline="") as handle:
        return pd.DataFrame(list(csv.DictReader(handle)))
