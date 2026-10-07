from bert_score import score as bert_score
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

POSITIVE_KW = ["네.", "네", "예", "yes", "가능", "허용", "좋", "맞", "그럼", "물론"]


def normalize_yesno(text: str, extra_positive=()) -> str:
    """응답을 '네.' 또는 '아니요.'로 통일 (긍정 키워드가 없으면 '아니요.')"""
    if not text:
        return "[빈 응답]"
    t = text.strip().replace(" ", "").lower()
    if any(k in t for k in [*POSITIVE_KW, *extra_positive]):
        return "네."
    return "아니요."


def compute_metrics(preds, gt):
    y_true = [1 if x.startswith("네") else 0 for x in gt]
    y_pred = [1 if x.startswith("네") else 0 for x in preds]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    smoothie = SmoothingFunction().method1
    bleu = sum(sentence_bleu([g], p, smoothing_function=smoothie)
               for g, p in zip(gt, preds)) / max(1, len(gt))
    P, R, F = bert_score(preds, gt, lang="ko", verbose=False)
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "mcc": mcc, "bleu": bleu, "bertscore": F.mean().item()
    }
