from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
import numpy as np

def compute_classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }

def purity_score(y_true, y_probs):
    target = []
    for yt, yp in zip(y_true, y_probs):
        if yt == 2:
            target.append(float(yp[0]))
    if not target:
        return 0.0
    return float(np.mean(target))
