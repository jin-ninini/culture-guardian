"""Hybrid RAG vs Non-RAG evaluation on the yes/no etiquette dataset."""
import json
import os

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from src.evaluation.llm_clients import PROVIDERS
from src.evaluation.metrics import compute_metrics, normalize_yesno
from src.rag.retriever import search_with_priority


def _answer(provider, question, context, use_rag, label):
    build_prompt, call, extra_positive, _ = PROVIDERS[provider]
    sys_p, usr_p = build_prompt(question, context, use_rag=use_rag)
    raw = call(sys_p, usr_p)
    norm = normalize_yesno(raw, extra_positive)
    print(f"\n[{label}] Q: {question}\n→ Raw: {raw}\n→ Norm: {norm}")
    return norm


def rag_answer(provider, question, country):
    results = search_with_priority(question, {"country": country})
    context = "\n\n".join([r["text"] for r in results[:5]]) if results else "(관련 문화 정보 없음)"
    return _answer(provider, question, context, True, "RAG")


def non_rag_answer(provider, question):
    return _answer(provider, question, None, False, "Non-RAG")


def evaluate(provider, csv_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nAPI 연결 상태 ({provider})")
    for keys in PROVIDERS[provider][3]:
        print(f"  {keys}: {'OK' if any(os.getenv(k) for k in keys.split('|')) else 'MISSING'}")

    df = pd.read_csv(csv_path).dropna(subset=["question", "answer"])
    gt, rag_preds, non_preds = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=provider.upper()):
        q, c, a = row["question"], row.get("country", ""), str(row["answer"]).strip()
        rag_preds.append(rag_answer(provider, q, c))
        non_preds.append(non_rag_answer(provider, q))
        gt.append(a)

    rag_m = compute_metrics(rag_preds, gt)
    non_m = compute_metrics(non_preds, gt)

    rows = [
        [name, f"{rag_m[key]:.4f}", f"{non_m[key]:.4f}"]
        for name, key in [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                          ("F1", "f1"), ("MCC", "mcc"), ("BLEU", "bleu"), ("BERTScore", "bertscore")]
    ]
    print(f"\n[{provider.upper()} 결과]")
    print(tabulate(rows, headers=["Metric", "Hybrid RAG", "Non-RAG"], tablefmt="grid"))

    json_out = os.path.join(out_dir, f"{provider}_hybrid_metrics.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({provider: {"hybrid_rag": rag_m, "non_rag": non_m}}, f, ensure_ascii=False, indent=2)

    csv_out = os.path.join(out_dir, f"eval_full_results_{provider}.csv")
    pd.DataFrame({
        "question": df["question"],
        "true_answer": gt,
        f"{provider}_RAG_answer": rag_preds,
        f"{provider}_NonRAG_answer": non_preds,
    }).to_csv(csv_out, index=False, encoding="utf-8-sig")

    print(f"\nJSON 저장 → {json_out}")
    print(f"CSV  저장 → {csv_out}")
