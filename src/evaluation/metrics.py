"""
Comprehensive evaluation metrics for veracity classification and source localization.
Computes Accuracy, Precision, Recall, F1, ROC-AUC, Hit@K, MRR, and hop distances.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import torch


def compute_classification_metrics(
    y_true: Union[List[int], np.ndarray, torch.Tensor],
    y_pred: Union[List[int], np.ndarray, torch.Tensor],
    y_probs: Optional[Union[List[float], np.ndarray, torch.Tensor]] = None,
) -> Dict[str, float]:
    """
    Computes full suite of classification metrics:
    Accuracy, Macro Precision/Recall/F1, and ROC-AUC.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if y_probs is not None and isinstance(y_probs, torch.Tensor):
        y_probs = y_probs.detach().cpu().numpy()

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Binary specific
    f1_binary = float(f1_score(y_true, y_pred, average="binary", zero_division=0))

    auc = 0.5
    if y_probs is not None:
        try:
            if len(np.unique(y_true)) > 1:
                # If y_probs is [N, 2], extract positive class
                if len(y_probs.shape) == 2 and y_probs.shape[1] == 2:
                    pos_probs = y_probs[:, 1]
                else:
                    pos_probs = y_probs
                auc = float(roc_auc_score(y_true, pos_probs))
        except Exception:
            auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn = float(cm[0, 0]) if cm.shape == (2, 2) else 0.0
    fp = float(cm[0, 1]) if cm.shape == (2, 2) else 0.0
    fn = float(cm[1, 0]) if cm.shape == (2, 2) else 0.0
    tp = float(cm[1, 1]) if cm.shape == (2, 2) else 0.0

    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "f1_binary": f1_binary,
        "roc_auc": auc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def aggregate_source_localization_metrics(eval_results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregates per-cascade source localization results across a test set:
    - Mean Hit@1, Hit@3, Hit@5
    - Mean Reciprocal Rank (MRR)
    - Mean Error Hop Distance
    """
    if not eval_results:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0, "mean_error_distance": 0.0}

    hit1 = float(np.mean([res.get("hit@1", 0.0) for res in eval_results]))
    hit3 = float(np.mean([res.get("hit@3", 0.0) for res in eval_results]))
    hit5 = float(np.mean([res.get("hit@5", 0.0) for res in eval_results]))
    mrr = float(np.mean([res.get("mrr", 0.0) for res in eval_results]))
    mean_dist = float(np.mean([res.get("error_distance", 0.0) for res in eval_results]))

    return {
        "hit@1": hit1,
        "hit@3": hit3,
        "hit@5": hit5,
        "mrr": mrr,
        "mean_error_distance": mean_dist,
        "num_evaluated_cascades": len(eval_results),
    }
