import os

import torch
from transformers import AutoTokenizer

from src.config import experiment_dir, load_config
from src.intent_classifier.model import IntentClassifier

LABEL_MAP = {0: "정보/호기심", 1: "비하/도발", 2: "편견"}


def load_intent_model(config_path: str):
    """Load tokenizer, trained model, device and config for an experiment."""
    cfg = load_config(config_path)
    model_path = os.path.join(experiment_dir(cfg), "best_model.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        use_fast=False,
        trust_remote_code=True
    )

    model = IntentClassifier(
        cfg["model_name"],
        num_labels=int(cfg["num_labels"]),
        dropout=float(cfg.get("dropout", 0.1)),
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다. 먼저 train.py를 실행하세요.\n"
            f"찾은 경로: {model_path}"
        )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return tokenizer, model, device, cfg


def predict_intent(text: str, tokenizer, model, device, config):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=config["max_length"],
        return_tensors="pt",
    )

    if "kobert" in config["model_name"].lower():
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])

    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        logits = out["logits"]
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        pred = int(torch.argmax(probs).item())

    return {
        "input": text,
        "intent": LABEL_MAP.get(pred, "unknown"),
        "probabilities": {
            LABEL_MAP[0]: round(float(probs[0]), 4),
            LABEL_MAP[1]: round(float(probs[1]), 4),
            LABEL_MAP[2]: round(float(probs[2]), 4),
        },
    }
