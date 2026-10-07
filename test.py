"""Evaluate the trained intent classifier on the test split and save error analysis."""
import argparse

from src.config import DEFAULT_INTENT_CONFIG, experiment_dir
from src.intent_classifier.evaluator import evaluate_and_save, save_error_analysis
from src.intent_classifier.inference import load_intent_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=DEFAULT_INTENT_CONFIG)
    args = parser.parse_args()

    tokenizer, model, device, cfg = load_intent_model(args.config)
    save_dir = experiment_dir(cfg)
    _, texts, labels, preds, _ = evaluate_and_save("test", model, tokenizer, device, cfg, save_dir)
    save_error_analysis(texts, labels, preds, cfg["labels"], save_dir)


if __name__ == "__main__":
    main()
