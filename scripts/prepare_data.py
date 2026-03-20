from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_pipeline import prepare_all_data


if __name__ == "__main__":
    competencies, jobs = prepare_all_data()
    print(f"Competencies prepared: {len(competencies)}")
    print(f"Jobs prepared: {len(jobs)}")
