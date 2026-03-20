from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_pipeline import load_reference_data, load_sample_profiles, prepare_all_data
from src.recommender import analyse_submission
from src.semantic_engine import SemanticEngine


def main() -> None:
    prepare_all_data()
    reference_df, jobs_df = load_reference_data()
    sample_profiles = load_sample_profiles()

    sample = sample_profiles[0]
    engine = SemanticEngine(backend="tfidf")
    results = analyse_submission(sample, reference_df, jobs_df, engine)

    print("=== DEMO AISCA ===")
    print(f"Candidate: {sample['candidate_name']}")
    print(f"Semantic backend: {engine.info().backend}")
    print(f"Final score: {results['final_score']:.2f}")
    print("Top jobs:")
    for _, row in results["top_jobs"].iterrows():
        print(f"- {row['job_title']} -> {row['final_score']:.2f}")
    print("Block scores:")
    for _, row in results["block_scores"].iterrows():
        print(f"- {row['block_name']} -> {row['block_score']:.2f}")


if __name__ == "__main__":
    main()
