from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_auc(
    y_true: np.ndarray,
    scores: Any,
    class_count: int,
) -> float | None:
    if scores is None:
        return None

    try:
        if class_count == 2:
            return float(roc_auc_score(y_true, scores[:, 1]))
        return float(
            roc_auc_score(
                y_true,
                scores,
                multi_class="ovr",
                average="weighted",
            )
        )
    except ValueError:
        return None


def summarize_predictions(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    scores: Any,
    class_count: int,
) -> dict[str, float | str]:
    auc_score = calculate_auc(y_true, scores, class_count)
    return {
        "Accuracy": float(accuracy_score(y_true, y_predicted)),
        "AUC Score": auc_score if auc_score is not None else "Not available",
        "Precision (weighted)": float(
            precision_score(
                y_true,
                y_predicted,
                average="weighted",
                zero_division=0,
            )
        ),
        "Recall (weighted)": float(
            recall_score(
                y_true,
                y_predicted,
                average="weighted",
                zero_division=0,
            )
        ),
        "F1 (weighted)": float(
            f1_score(
                y_true,
                y_predicted,
                average="weighted",
                zero_division=0,
            )
        ),
        "MCC": float(matthews_corrcoef(y_true, y_predicted)),
    }


def confusion_table(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    return confusion_matrix(
        y_true,
        y_predicted,
        labels=list(range(len(labels))),
    )


def build_classification_report(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    class_names: np.ndarray,
) -> str:
    labels = list(range(len(class_names)))
    return classification_report(
        y_true,
        y_predicted,
        labels=labels,
        target_names=class_names.astype(str),
        zero_division=0,
    )
