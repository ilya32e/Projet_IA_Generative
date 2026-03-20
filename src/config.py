from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SUBMISSIONS_DIR = PROCESSED_DIR / "submissions"
CACHE_DIR = BASE_DIR / "cache"

COMPETENCY_RAW_PATH = RAW_DIR / "competency_reference_raw.csv"
JOB_RAW_PATH = RAW_DIR / "job_profiles_raw.csv"
SAMPLE_PROFILES_PATH = RAW_DIR / "sample_profiles.json"

COMPETENCY_PROCESSED_PATH = PROCESSED_DIR / "competency_reference.csv"
JOB_PROCESSED_PATH = PROCESSED_DIR / "job_profiles.csv"
SUBMISSION_INDEX_PATH = PROCESSED_DIR / "submission_index.csv"

GENAI_CACHE_PATH = CACHE_DIR / "genai_cache.json"
GENAI_LOG_PATH = CACHE_DIR / "genai_log.csv"

SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
LOCAL_GENAI_MODEL_NAME = os.getenv("LOCAL_GENAI_MODEL_NAME", "google/flan-t5-small")
GENAI_MODEL_NAME = os.getenv("GENAI_MODEL_NAME", GEMINI_MODEL_NAME)
GENAI_PROVIDER = os.getenv("GENAI_PROVIDER", "gemini")
DEFAULT_BACKEND = os.getenv("SEMANTIC_BACKEND", "auto")
SEMANTIC_THRESHOLD = float(os.getenv("SEMANTIC_THRESHOLD", "0.52"))
TOP_N_JOBS = int(os.getenv("TOP_N_JOBS", "3"))

BLOCK_NAME_MAP = {
    "B01": "Preparation des donnees",
    "B02": "Visualisation et tableau de bord",
    "B03": "Analyse exploratoire",
    "B04": "NLP semantique",
    "B05": "IA generative et RAG",
}

LEVEL_LABELS = {
    0: "Aucun niveau",
    1: "Notions",
    2: "Debutant",
    3: "Intermediaire",
    4: "Autonome",
    5: "Avance",
}


def ensure_directories() -> None:
    for path in (RAW_DIR, PROCESSED_DIR, SUBMISSIONS_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)
