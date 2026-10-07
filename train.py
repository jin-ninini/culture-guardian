"""Train the intent classifier and save the best checkpoint to models/intent_classifier/<experiment>/."""
import argparse

from src.config import DEFAULT_INTENT_CONFIG, load_config
from src.intent_classifier.trainer import IntentTrainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=DEFAULT_INTENT_CONFIG)
    args = parser.parse_args()

    trainer = IntentTrainer(load_config(args.config))
    trainer.train()


if __name__ == "__main__":
    main()
