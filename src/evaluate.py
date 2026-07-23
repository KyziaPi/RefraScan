# src/evaluate.py
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_model(
    model,
    test_ds,
    steps=None,
    y_true=None,
    class_names=["Emmetropia", "Myopia", "Hyperopia"],
):
  """Evaluates a trained model on a dataset or infinite generator.

  Args:
      model: Trained tf.keras.Model.
      test_ds: Generator or dataset.
      steps (int): Number of batch steps to evaluate (required for infinite
        generators).
      y_true (array-like): Ground truth labels. If None, extracts them directly
        from test_ds.
      class_names (list): Target class names.
  """
  print("\n--- Running Inference ---")

  # 1. Run prediction with explicit step limit
  y_pred_probs = model.predict(test_ds, steps=steps)
  class_multipliers = np.array([2.5, 1.0, 1.2])  # [Emmetropia, Myopia, Hyperopia]
  adjusted_probs = y_pred_probs * class_multipliers
  y_pred = np.argmax(adjusted_probs, axis=1)

  # 2. Extract ground truth labels safely
  if y_true is None:
    if hasattr(test_ds, "labels"):
      y_true = np.array(test_ds.labels)
    elif hasattr(test_ds, "df") and "classification_encoded" in test_ds.df:
      y_true = test_ds.df["classification_encoded"].values
    else:
      y_true_list = []
      for idx, batch in enumerate(test_ds):
        if steps is not None and idx >= steps:
          break
        batch_y = batch[1]
        y_true_list.extend(batch_y)
      y_true = np.array(y_true_list)

  # Align lengths if truncation occurred
  min_len = min(len(y_true), len(y_pred))
  y_true = y_true[:min_len]
  y_pred = y_pred[:min_len]

  # 3. Compute Metrics
  accuracy = accuracy_score(y_true, y_pred)
  precision, recall, f1, _ = precision_recall_fscore_support(
      y_true, y_pred, average="weighted"
  )
  cm = confusion_matrix(y_true, y_pred)

  
  print("\n--- Classification Report ---")
  print(classification_report(y_true, y_pred, target_names=class_names))
  print(f"\n--- Evaluation Metrics ---")
  print(f"\nAccuracy         : {accuracy:.4f} ({accuracy * 100:.2f}%)")
  print(f"Precision (Weighted): {precision:.4f}")
  print(f"Recall (Weighted)   : {recall:.4f}")
  print(f"F1 Score (Weighted) : {f1:.4f}")

  return {
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1": f1,
      "confusion_matrix": cm,
  }