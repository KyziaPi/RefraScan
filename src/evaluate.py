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
      steps (int): Number of batch steps to evaluate.
      y_true (array-like): Ground truth labels.
      class_names (list): Target class names (List of strings).
  """
  print("\n--- Running Inference ---")

  # 1. Run prediction
  y_pred_probs = model.predict(test_ds, steps=steps)
  y_pred = np.argmax(y_pred_probs, axis=1)

  # 2. Extract ground truth labels safely
  if y_true is None:
    if hasattr(test_ds, "labels"):
      y_true = np.array(test_ds.labels)
    elif hasattr(test_ds, "df") and "classification_encoded" in test_ds.df:
      y_true = test_ds.df["classification_encoded"].values
    else:
      # Warning: If test_ds is a generator, pass y_true explicitly to prevent batch desync
      y_true_list = []
      for idx, batch in enumerate(test_ds):
        if steps is not None and idx >= steps:
          break
        batch_y = batch[1]
        y_true_list.extend(batch_y)
      y_true = np.array(y_true_list)

  # 3. Align lengths FIRST to prevent IndexErrors
  min_len = min(len(y_true), len(y_pred_probs))
  y_true = y_true[:min_len]
  y_pred = y_pred[:min_len]
  y_pred_probs = y_pred_probs[:min_len]

  # Create a lookup dictionary for debug printing without mutating the list class_names
  class_dict = {i: name for i, name in enumerate(class_names)}

  # 4. Print Raw Probabilities (First 10 Images)
  print("\n--- Raw Probabilities: First 10 Images ---")
  print(
      f"{'True Label':<12} | {'Emmetropia':<12} | {'Myopia':<12} |"
      " {'Hyperopia':<12}"
  )
  print("-" * 55)

  for i in range(min(10, len(y_true))):
    true_label = class_dict.get(y_true[i], str(y_true[i]))
    p_emm, p_myo, p_hyp = y_pred_probs[i]
    print(
        f"{true_label:<12} | {p_emm:.4f}       | {p_myo:.4f}      "
        f" | {p_hyp:.4f}"
    )

  # 5. Specifically isolate and check ACTUAL Emmetropia images
  print("\n--- Raw Probabilities: ACTUAL Emmetropia Images ---")
  print(
      f"{'True Label':<12} | {'Emmetropia':<12} | {'Myopia':<12} |"
      " {'Hyperopia':<12}"
  )
  print("-" * 55)

  emmetropia_indices = np.where(y_true == 0)[0]
  for i in emmetropia_indices[:10]:
    true_label = class_dict.get(0, "Emmetropia")
    p_emm, p_myo, p_hyp = y_pred_probs[i]
    marker = " <-- CLOSE!" if p_emm > 0.20 else ""
    print(
        f"{true_label:<12} | {p_emm:.4f}       | {p_myo:.4f}      "
        f" | {p_hyp:.4f}{marker}"
    )

  # 6. Compute Metrics
  accuracy = accuracy_score(y_true, y_pred)
  precision, recall, f1, _ = precision_recall_fscore_support(
      y_true, y_pred, average="weighted"
  )
  cm = confusion_matrix(y_true, y_pred)

  print("\n--- Classification Report ---")
  # class_names remains a list of strings, satisfying sklearn's requirements
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