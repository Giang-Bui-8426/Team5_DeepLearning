"""Classification metrics and lossless, atomic artifact IO shared by train/test."""
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    temp.replace(path)


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(clean(rows))


def classification_metrics(labels, probabilities, loss):
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    if (y.ndim != 1 or not len(y) or not np.issubdtype(y.dtype, np.integer)
            or np.any((y < 0) | (y > 9)) or p.shape != (len(y), 10)
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
            or not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0) or not math.isfinite(loss)):
        raise ValueError('Invalid labels/probabilities/loss')
    cm = confusion_matrix(y, p.argmax(1), labels=np.arange(10))
    precision, recall, f1, support = precision_recall_fscore_support(
        y, p.argmax(1), labels=np.arange(10), zero_division=0)
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    tn = len(y) - cm.sum(0) - cm.sum(1) + tp
    fpr = [float(fp[k] / (fp[k] + tn[k])) if fp[k] + tn[k] else None for k in range(10)]
    aucs, rocs = [], []
    grid = np.linspace(0, 1, 201)
    for k in range(10):
        binary = y == k
        if 0 < support[k] < len(y):
            aucs.append(float(roc_auc_score(binary, p[:, k])))
            x, tpr, _ = roc_curve(binary, p[:, k])
            rocs.append(np.interp(grid, x, tpr).tolist())
        else:
            aucs.append(None)
            rocs.append(None)
    valid_fpr = [v for v in fpr if v is not None]
    result = dict(samples=len(y), correct_count=int(tp.sum()), incorrect_count=int(len(y)-tp.sum()),
                  loss=float(loss), accuracy=float(tp.sum()/len(y)), precision_macro=float(precision.mean()),
                  recall_macro=float(recall.mean()), f1_macro=float(f1.mean()),
                  fpr_macro=float(np.mean(valid_fpr)) if valid_fpr else None,
                  auc_macro_ovr=float(np.mean(aucs)) if all(v is not None for v in aucs) else None,
                  auc_micro_ovr=float(roc_auc_score(np.eye(10)[y].ravel(), p.ravel())))
    per_class = [dict(class_id=k, precision=float(precision[k]), recall=float(recall[k]), f1=float(f1[k]),
                      support=int(support[k]), fpr=fpr[k], auc=aucs[k]) for k in range(10)]
    return dict(metrics=result, cm=cm.tolist(), per_class=per_class,
                roc=dict(grid=grid.tolist(), tpr=rocs, auc=aucs,
                         macro=np.mean(rocs, axis=0).tolist() if all(r is not None for r in rocs) else None),
                undefined_classes=[k for k, a in enumerate(aucs) if a is None])
