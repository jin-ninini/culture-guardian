"""Compare Hybrid RAG vs Non-RAG answers of an LLM on data/eval/eval_dataset.csv.

Results are written to data/eval/results/. Requires the vector DB (build_vectordb.py)
and the API key of the chosen provider in .env.
"""
import argparse
import os

from src.config import DATA_DIR
from src.evaluation.llm_clients import PROVIDERS
from src.evaluation.runner import evaluate

CSV_PATH = os.path.join(DATA_DIR, "eval", "eval_dataset.csv")
OUT_DIR = os.path.join(DATA_DIR, "eval", "results")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=[*PROVIDERS, "all"], required=True)
    args = parser.parse_args()

    for provider in (PROVIDERS if args.model == "all" else [args.model]):
        evaluate(provider, CSV_PATH, OUT_DIR)


if __name__ == "__main__":
    main()
