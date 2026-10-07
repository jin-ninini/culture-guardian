"""Evaluate the trained intent classifier on the validation split."""
import argparse

from src.config import DEFAULT_INTENT_CONFIG, experiment_dir
from src.intent_classifier.evaluator import evaluate_and_save
from src.intent_classifier.inference import load_intent_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=DEFAULT_INTENT_CONFIG)
    args = parser.parse_args()

    tokenizer, model, device, cfg = load_intent_model(args.config)
    evaluate_and_save("valid", model, tokenizer, device, cfg, experiment_dir(cfg))


if __name__ == "__main__":
    main()
