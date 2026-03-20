import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.data_pipeline import ensure_submission_identity, load_reference_data, load_sample_profiles, prepare_all_data
from src.genai import GenerationSettings, LocalGenAI
from src.recommender import analyse_submission
from src.semantic_engine import SemanticEngine


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        prepare_all_data()
        cls.reference_df, cls.jobs_df = load_reference_data()
        cls.sample = load_sample_profiles()[0]

    def test_processed_reference_has_no_duplicate_clean_text(self) -> None:
        clean_keys = self.reference_df["block_id"] + "__" + self.reference_df["competency_slug"]
        self.assertEqual(len(clean_keys), clean_keys.nunique())

    def test_analysis_returns_top_three_jobs(self) -> None:
        engine = SemanticEngine(backend="tfidf")
        results = analyse_submission(self.sample, self.reference_df, self.jobs_df, engine)
        self.assertEqual(len(results["top_jobs"]), 3)
        self.assertGreaterEqual(results["final_score"], 0.0)
        self.assertLessEqual(results["final_score"], 1.0)

    def test_genai_cache_is_reused(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            generator = LocalGenAI(
                model_name="invalid-local-model",
                cache_path=tmp_path / "cache.json",
                log_path=tmp_path / "log.csv",
                allow_model_loading=False,
            )
            context = {
                "candidate_name": "Test",
                "target_role": "BI Analyst",
                "top_jobs": [{"job_title": "BI Analyst", "final_score": 0.72}],
                "block_scores": [{"block_name": "Visualisation et tableau de bord", "block_score": 0.68}],
                "strengths": ["Construire un tableau de bord clair"],
                "gaps": ["Construire un contexte RAG"],
                "overall_score": 0.66,
            }
            first = generator.generate("bio", context, GenerationSettings(max_new_tokens=70, temperature=0.1))
            second = generator.generate("bio", context, GenerationSettings(max_new_tokens=70, temperature=0.1))
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])

    def test_generate_once_locks_one_call_per_request_key(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            generator = LocalGenAI(
                model_name="invalid-local-model",
                cache_path=tmp_path / "cache.json",
                log_path=tmp_path / "log.csv",
                allow_model_loading=False,
            )
            context = {
                "candidate_name": "Test",
                "target_role": "BI Analyst",
                "top_jobs": [{"job_title": "BI Analyst", "final_score": 0.72}],
                "block_scores": [{"block_name": "Visualisation et tableau de bord", "block_score": 0.68}],
                "strengths": ["Construire un tableau de bord clair"],
                "gaps": ["Construire un contexte RAG"],
                "overall_score": 0.66,
            }
            first = generator.generate_once("plan", context, "request_1", GenerationSettings(max_new_tokens=70, temperature=0.1))
            second = generator.generate_once("plan", context, "request_1", GenerationSettings(max_new_tokens=200, temperature=0.7))
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])

    def test_submission_identity_is_stable_for_same_payload(self) -> None:
        submission = {
            "candidate_name": "Profil etudiant",
            "target_role": "BI Analyst",
            "experience_months": 8,
            "tools": ["Python", "Pandas"],
            "focus_blocks": ["Preparation des donnees"],
            "project_focus": "Analyse de donnees",
            "tokenization_used": "Non",
            "levels": {"python": 4},
            "project_text": "Nettoyage de donnees",
            "dashboard_text": "Dashboard simple",
            "genai_text": "Bio courte",
        }
        first = ensure_submission_identity(submission)
        second = ensure_submission_identity(submission)
        self.assertEqual(first["submission_id"], second["submission_id"])

    def test_ranking_does_not_depend_on_target_role(self) -> None:
        engine = SemanticEngine(backend="tfidf")
        submission_a = dict(self.sample)
        submission_b = dict(self.sample)
        submission_a["target_role"] = "BI Analyst"
        submission_b["target_role"] = "Analyste NLP"
        results_a = analyse_submission(submission_a, self.reference_df, self.jobs_df, engine)
        results_b = analyse_submission(submission_b, self.reference_df, self.jobs_df, engine)
        self.assertListEqual(
            results_a["top_jobs"]["job_title"].tolist(),
            results_b["top_jobs"]["job_title"].tolist(),
        )


if __name__ == "__main__":
    unittest.main()
