"""Balayage des parametres GenAI : temperature x top_p (C5.3).

Pour un contexte candidat representatif, genere une bio sur une grille de
(temperature, top_p), score chaque sortie avec la grille d'evaluation existante
(`evaluate_generated_text`), et ecrit reports/param_sweep.csv.

Objectif : demontrer l'effet des parametres d'echantillonnage sur la qualite et la
diversite des sorties, et justifier le reglage retenu en production.

Usage : python scripts/param_sweep.py

Le sweep utilise un cache temporaire isole : il genere donc reellement a chaque
combinaison (sans polluer le cache applicatif). Selon l'environnement, le backend
peut etre Gemini, le modele local (flan-t5) ou le template de repli.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics import evaluate_generated_text  # noqa: E402
from src.genai import GenerationSettings, LocalGenAI  # noqa: E402

REPORTS_DIR = ROOT / "reports"

# Grille de parametres a explorer
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]
TOP_PS = [0.80, 0.95, 1.00]

# Contexte candidat representatif (fixe pour isoler l'effet des parametres)
CONTEXT = {
    "candidate_name": "Lina Martin",
    "target_role": "BI Analyst",
    "overall_score": 0.71,
    "top_jobs": [
        {"job_title": "BI Analyst", "final_score": 0.74},
        {"job_title": "Data Analyst Junior", "final_score": 0.69},
        {"job_title": "Data Quality Analyst", "final_score": 0.66},
    ],
    "strengths": ["visualisation de donnees", "SQL", "communication"],
    "gaps": ["machine learning", "python avance", "data engineering"],
    "block_scores": [
        {"block_name": "Analyse de donnees", "block_score": 0.78},
        {"block_name": "Restitution", "block_score": 0.81},
        {"block_name": "Modelisation", "block_score": 0.52},
    ],
}

# Bornes de longueur attendues pour une bio (cf. prompt bio : 60-90 mots)
MIN_WORDS, MAX_WORDS = 60, 90


def context_terms() -> list[str]:
    terms = list(CONTEXT["strengths"]) + list(CONTEXT["gaps"])
    terms += [job["job_title"] for job in CONTEXT["top_jobs"]]
    return terms


def main() -> int:
    terms = context_terms()
    rows = []
    backend_modes = set()

    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_path = Path(tmp_dir) / "sweep_cache.json"
        log_path = Path(tmp_dir) / "sweep_log.csv"

        print("Balayage temperature x top_p sur la generation de bio\n")
        for temperature in TEMPERATURES:
            for top_p in TOP_PS:
                engine = LocalGenAI(cache_path=cache_path, log_path=log_path)
                settings = GenerationSettings(max_new_tokens=140, temperature=temperature, top_p=top_p)
                result = engine.generate("bio", CONTEXT, settings=settings)
                metrics = evaluate_generated_text(result["text"], terms, MIN_WORDS, MAX_WORDS)
                backend_modes.add(result["mode"])
                rows.append({
                    "temperature": temperature,
                    "top_p": top_p,
                    "mode": result["mode"],
                    "word_count": int(metrics["word_count"]),
                    "coverage": metrics["coverage"],
                    "readability": metrics["readability"],
                    "diversity": metrics["diversity"],
                    "overall": metrics["overall"],
                })
                print(f"  T={temperature:<3} top_p={top_p:<4}  "
                      f"overall={metrics['overall']:.3f}  coverage={metrics['coverage']:.3f}  "
                      f"diversity={metrics['diversity']:.3f}  words={int(metrics['word_count'])}  [{result['mode']}]")

    report = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORTS_DIR / "param_sweep.csv"
    report.to_csv(output, index=False)

    best = report.sort_values("overall", ascending=False).iloc[0]
    print(f"\nRapport ecrit : {output}")
    print(f"Backends utilises : {', '.join(sorted(backend_modes))}")
    print(f"Meilleur reglage : temperature={best['temperature']} / top_p={best['top_p']} "
          f"(overall={best['overall']:.3f})")
    if backend_modes == {"template_fallback"}:
        print("\nNote : backend en repli template (deterministe) -> les sorties ne varient pas. "
              "Avec Gemini ou le modele local, la diversite augmente avec temperature/top_p.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
