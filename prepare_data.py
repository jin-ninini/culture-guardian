"""Prepare intent classifier splits.

split : data/intent/intent_dataset_final.csv → data/intent/splits/{train,valid,test}.csv
clean : drop 편견 ↔ 비하/도발 confusions found by test.py → data/intent/splits/*_clean.csv
"""
import argparse
import os

from src.config import DATA_DIR, DEFAULT_INTENT_CONFIG, experiment_dir, load_config
from src.intent_classifier.data_preparation import remove_confusing_samples, split_dataset

SRC_PATH = os.path.join(DATA_DIR, "intent", "intent_dataset_final.csv")
SPLITS_DIR = os.path.join(DATA_DIR, "intent", "splits")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["split", "clean"])
    parser.add_argument("--config", type=str, default=DEFAULT_INTENT_CONFIG,
                        help="experiment whose misclassified_samples.csv is used by 'clean'")
    args = parser.parse_args()

    if args.step == "split":
        split_dataset(SRC_PATH, SPLITS_DIR)
    else:
        cfg = load_config(args.config)
        remove_confusing_samples(os.path.join(experiment_dir(cfg), "misclassified_samples.csv"), SPLITS_DIR)


if __name__ == "__main__":
    main()
