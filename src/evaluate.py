import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_model(
    model, test_ds, class_names=["Emmetropia", "Myopia", "Hyperopia"]
):
  """Evaluates a trained multimodal Keras model on a test dataset/generator.

  Args:
      model: Trained tf.keras.Model instance.
      test_ds: Test DataGenerator or tf.data.Dataset yielding ([inputs],
        labels).
      class_names (list): List of class string names for target encoding mapping.

  Returns:
      dict: Dictionary containing Accuracy, Precision, Recall, F1, and Confusion
      Matrix.
  """
  print("\n--- Running Inference on Test Dataset ---")

  # 1. Generate predicted probabilities
  y_pred_probs = model.predict(test_ds)
  y_pred = np.argmax(y_pred_probs, axis=1)

  # 2. Safely extract ground truth true labels (y_true)
  if hasattr(test_ds, "labels"):
    # If custom generator stores labels in .labels
    y_true = np.array(test_ds.labels)
  elif hasattr(test_ds, "df") and "classification_encoded" in test_ds.df:
    # If custom generator holds underlying DataFrame
    y_true = test_ds.df["classification_encoded"].values
  else:
    # Fallback: iterate directly through generator batches
    y_true_list = []
    for _, batch_y in test_ds:
      y_true_list.extend(
          batch_y.numpy() if hasattr(batch_y, "numpy") else batch_y
      )
    y_true = np.array(y_true_list)

  # Match array lengths in case generator truncated final batch
  if len(y_true) != len(y_pred):
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

  # 3. Calculate metrics
  accuracy = accuracy_score(y_true, y_pred)
  precision, recall, f1, _ = precision_recall_fscore_support(
      y_true, y_pred, average="weighted"
  )
  cm = confusion_matrix(y_true, y_pred)

  # 4. Display formatted summary
  print("\n" + "=" * 45)
  print("          EVALUATION METRICS SUMMARY          ")
  print("=" * 45)
  print(f"Accuracy         : {accuracy:.4f} ({accuracy * 100:.2f}%)")
  print(f"Precision (Weighted): {precision:.4f}")
  print(f"Recall (Weighted)   : {recall:.4f}")
  print(f"F1 Score (Weighted) : {f1:.4f}")
  print("-" * 45)
  print("Confusion Matrix:")
  print(cm)
  print("-" * 45)

  # Print detailed breakdown per class
  print("\nDetailed Per-Class Classification Report:")
  print(classification_report(y_true, y_pred, target_names=class_names))

  # 5. Return metrics
  return {
      "accuracy": accuracy,
      "precision": precision,
      "recall": recall,
      "f1": f1,
      "confusion_matrix": cm,
  }