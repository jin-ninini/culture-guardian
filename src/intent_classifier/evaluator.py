import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tqdm import tqdm

from src.intent_classifier.metrics import compute_classification_metrics, purity_score


@torch.no_grad()
def evaluate_split(model, tokenizer, path, max_length, is_kobert, device):
    df = pd.read_csv(path)
    texts, labels = df["text"].tolist(), df["label"].tolist()

    preds, probs_all = [], []
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    for text, label in tqdm(zip(texts, labels), total=len(texts), desc=f"Evaluating {os.path.basename(path)}"):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        if is_kobert:
            enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = model(**enc)
        logits = outputs["logits"]
        loss = loss_fn(logits, torch.tensor([label], device=device))
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1)
        preds.append(torch.argmax(probs, dim=-1).item())
        probs_all.append(probs.squeeze(0).cpu().tolist())

    metrics = compute_classification_metrics(labels, preds)
    metrics["purity"] = purity_score(labels, probs_all)
    metrics["loss"] = total_loss / len(df)

    return metrics, texts, labels, preds, probs_all


def evaluate_and_save(split_name, model, tokenizer, device, config, save_dir):
    """Evaluate one split, print metrics and save them as <split>_metrics.json."""
    metrics, texts, labels, preds, probs = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        path=config[f"{split_name}_path"],
        max_length=config["max_length"],
        is_kobert="kobert" in config["model_name"].lower(),
        device=device,
    )
    with open(os.path.join(save_dir, f"{split_name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n{split_name} 평가 결과:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics, texts, labels, preds, probs


def save_error_analysis(texts, labels, preds, label_names, save_dir):
    """Confusion matrix, classification report and misclassified samples."""
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(cm_path)
    print(f"[Saved] 혼동행렬 → {cm_path}")

    report = classification_report(labels, preds, target_names=label_names, digits=4)
    print("\n===== Classification Report (test) =====")
    print(report)
    with open(os.path.join(save_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    wrong = [
        {"text": text, "true_label": label_names[yt], "pred_label": label_names[yp]}
        for text, yt, yp in zip(texts, labels, preds)
        if yt != yp
    ]
    wrong_df = pd.DataFrame(wrong, columns=["text", "true_label", "pred_label"])
    wrong_path = os.path.join(save_dir, "misclassified_samples.csv")
    wrong_df.to_csv(wrong_path, index=False, encoding="utf-8-sig")
    print(f"[Saved] 오분류 샘플 → {wrong_path}  (총 {len(wrong_df)}개)")
