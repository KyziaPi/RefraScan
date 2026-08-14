# src/evaluate.py
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CLASS_NAMES = ["Emmetropia", "Myopia", "Hyperopia"]


def predict_generator(model, generator, n_samples):
    """Collect exactly n_samples predictions from a finite generator."""
    y_prob = model.predict(generator, steps=int(np.ceil(n_samples / generator.batch_size)) if hasattr(generator, "batch_size") else None, verbose=0)
    return y_prob[:n_samples]


def evaluate_predictions(y_true, y_pred, class_names=CLASS_NAMES, verbose=True):
    """Compute the thesis-primary metrics and confusion matrix."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=np.arange(len(class_names))
        ),
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    result["classification_report"] = report

    for class_name in class_names:
        result[f"{class_name}_precision"] = report[class_name]["precision"]
        result[f"{class_name}_recall"] = report[class_name]["recall"]
        result[f"{class_name}_f1"] = report[class_name]["f1-score"]

    if verbose:
        print("\n--- Classification Report ---")
        print(
            classification_report(
                y_true,
                y_pred,
                labels=np.arange(len(class_names)),
                target_names=class_names,
                zero_division=0,
            )
        )
        print("--- Evaluation Metrics ---")
        print(f"Accuracy          : {result['accuracy']:.4f}")
        print(f"Balanced Accuracy : {result['balanced_accuracy']:.4f}")
        print(f"Macro Precision   : {result['macro_precision']:.4f}")
        print(f"Macro Recall      : {result['macro_recall']:.4f}")
        print(f"Macro F1          : {result['macro_f1']:.4f}")
        print("\nConfusion Matrix:")
        print(result["confusion_matrix"])

    return result


def evaluate_model(model, test_ds, y_true, steps=None, class_names=CLASS_NAMES):
    """Evaluate a model using a deterministic generator and explicit labels."""
    print("\n--- Running Inference ---")
    y_pred_probs = model.predict(test_ds, steps=steps, verbose=0)
    y_pred_probs = y_pred_probs[: len(y_true)]
    y_pred = np.argmax(y_pred_probs, axis=1)
    result = evaluate_predictions(y_true, y_pred, class_names=class_names)
    # Preserve predictions for downstream error analysis and Grad-CAM without
    # performing another holdout metric evaluation.
    result["y_true"] = np.asarray(y_true)[: len(y_pred)]
    result["y_pred"] = y_pred
    result["y_pred_probs"] = y_pred_probs
    return result


def metrics_to_row(result, fold=None, model_name=None):
    """Convert evaluation output into a compact DataFrame row."""
    row = {
        "Fold": fold,
        "Model": model_name,
        "Accuracy": result["accuracy"],
        "Balanced Accuracy": result["balanced_accuracy"],
        "Macro Precision": result["macro_precision"],
        "Macro Recall": result["macro_recall"],
        "Macro F1": result["macro_f1"],
    }
    for class_name in CLASS_NAMES:
        row[f"{class_name} Precision"] = result[f"{class_name}_precision"]
        row[f"{class_name} Recall"] = result[f"{class_name}_recall"]
        row[f"{class_name} F1"] = result[f"{class_name}_f1"]
    return row


def majority_class_baseline_metrics(y_true, class_names=CLASS_NAMES):
    """Evaluate a constant majority-class classifier as the study baseline."""
    y_true = np.asarray(y_true).astype(int)
    counts = np.bincount(y_true, minlength=len(class_names))
    majority_class = int(np.argmax(counts))
    y_pred = np.full_like(y_true, majority_class)
    result = evaluate_predictions(y_true, y_pred, class_names=class_names, verbose=False)
    result["majority_class"] = majority_class
    result["majority_class_name"] = class_names[majority_class]
    return result
