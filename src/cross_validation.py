import numpy as np
import pandas as pd
import math
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.preprocessing import (
    calculate_class_weights,
    create_multimodal_generator,
    load_and_preprocess_image,
    scale_age_feature,
)
from src.models import build_model
from src.train import train_model
from src.evaluate import evaluate_model

def split_holdout_test(
    df: pd.DataFrame,
    patient_col: str = "ID",
    target_col: str = "classification_encoded",
    test_size: float = 0.15,
    random_state: int = 42,
):
  """Splits dataframe into 85% CV pool and 15% Holdout Test Set strictly by patient ID."""
  df = df.copy()

  patient_classes = df.groupby(patient_col)[target_col].first()
  unique_ids = patient_classes.index.values
  unique_labels = patient_classes.values

  train_cv_ids, test_ids = train_test_split(
      unique_ids,
      test_size=test_size,
      stratify=unique_labels,
      random_state=random_state,
  )

  cv_df = df[df[patient_col].isin(train_cv_ids)].copy().reset_index(drop=True)
  test_df = df[df[patient_col].isin(test_ids)].copy().reset_index(drop=True)

  print(
      f"Dataset Split: {len(cv_df)} samples for 10-Fold CV | {len(test_df)}"
      f" samples in Holdout Test Set ({test_size * 100:.0f}%)"
  )
  return cv_df, test_df

def run_cross_validation(
    df: pd.DataFrame,
    model_name: str = "efficientnet",
    preprocess_input: callable = None,
    patient_col: str = "ID",
    target_col: str = "classification_encoded",
    n_splits: int = 10,
    batch_size: int = 16,
    epochs: int = 30,
    learning_rate: float = 0.0001,
    holdout_test_size: float = 0.15,
):
    """Executes 10-Fold CV with an isolated Holdout Test Set."""
    
    # 1. First, isolate 15% Holdout Test Set by Patient ID (No patient overlap)
    cv_df, holdout_test_df = split_holdout_test(
        df,
        patient_col=patient_col,
        target_col=target_col,
        test_size=holdout_test_size,
    )
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Tracking metrics
    val_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    test_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    print(f"\n{'='*65}")
    print(f"STARTING {n_splits}-FOLD CROSS VALIDATION")
    print(f"{'='*65}\n")

    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(cv_df, y=cv_df[target_col], groups=cv_df[patient_col])
    ):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        # 1. Split data for the current fold
        train_df = cv_df.iloc[train_idx].copy()
        val_df = cv_df.iloc[val_idx].copy()

        # 2. Prevent Data Leakage: Scale age strictly using the fold's training set
        train_df, val_df, current_test_df = scale_age_feature(
            train_df=train_df,
            val_df=val_df,
            test_df=holdout_test_df.copy(),
            age_col="age",
            scaler_save_path=f"scaler_fold_{fold + 1}.pkl"
        )
        
        metadata_cols = ["age_scaled"]

        # 3. Calculate fold-specific class weights
        class_weights = calculate_class_weights(train_df, target_col)

        # 4. Create Multimodal Generators
        train_gen = create_multimodal_generator(
            train_df, 
            metadata_cols, 
            class_weights,
            batch_size=batch_size, 
            augment=True, 
            preprocess_fn=preprocess_input,
            image_loader=load_and_preprocess_image
        )
        val_gen = create_multimodal_generator(
            val_df, 
            metadata_cols,
            class_weights=None,
            batch_size=batch_size, 
            augment=False, 
            preprocess_fn=preprocess_input,
            image_loader=load_and_preprocess_image
        )
        test_gen = create_multimodal_generator(
            current_test_df,
            metadata_cols,
            class_weights=None,
            batch_size=batch_size,
            augment=False,
            preprocess_fn=preprocess_input,
            image_loader=load_and_preprocess_image
        )
        
        # Calculate steps per epoch based on dataset lengths and batch size
        majority_class_count = train_df["classification_encoded"].value_counts().max()
        effective_train_size = majority_class_count * 3
        steps_per_epoch = int(np.ceil(effective_train_size / batch_size))
        validation_steps = math.ceil(len(val_df) / batch_size)
        test_steps = math.ceil(len(current_test_df) / batch_size)

        # 5. Build a fresh model for each fold
        model = build_model(
            model_name=model_name,
            input_image_shape=(300, 300, 3), 
            num_metadata_features=len(metadata_cols), 
            num_classes=3,
            dropout_rate=0.4
        )

        fold_model_path = f"best_{model_name}_fold_{fold + 1}.h5"
        

        # 6. Train the model
        train_model(
            model=model,
            train_ds=train_gen,
            val_ds=val_gen,
            epochs=epochs,
            learning_rate=learning_rate,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            save_path=fold_model_path,
            class_weight=class_weights
        )

        # 7. Evaluate the best model on the fold's validation set
        model.load_weights(fold_model_path)
        print("\n[Validation Set Evaluation]")
        v_res = evaluate_model(
            model=model,
            test_ds=val_gen,
            steps=validation_steps,
            y_true=val_df[target_col].values,
            class_names=["Emmetropia", "Myopia", "Hyperopia"],
        )
        
        # 8. Evaluate on Holdout Test Set
        print("\n[Holdout Test Set Evaluation]")
        t_res = evaluate_model(
            model=model,
            test_ds=test_gen,
            steps=test_steps,
            y_true=current_test_df[target_col].values,
            class_names=["Emmetropia", "Myopia", "Hyperopia"],
        )

        # 9. Store results
        for m in ["accuracy", "precision", "recall", "f1"]:
            val_metrics[m].append(v_res.get(m, 0))
            test_metrics[m].append(t_res.get(m, 0))
            
        # Build DataFrames for per-fold breakdown tables
        num_completed_folds = len(val_metrics["accuracy"])
        fold_names = [f"Fold {i+1}" for i in range(num_completed_folds)]

        val_summary_df = pd.DataFrame({
            "Fold": fold_names,
            "Accuracy": val_metrics["accuracy"],
            "Precision": val_metrics["precision"],
            "Recall": val_metrics["recall"],
            "F1 Score": val_metrics["f1"],
        })

        test_summary_df = pd.DataFrame({
            "Fold": fold_names,
            "Accuracy": test_metrics["accuracy"],
            "Precision": test_metrics["precision"],
            "Recall": test_metrics["recall"],
            "F1 Score": test_metrics["f1"],
        })

    # 10. Calculate and display final average metrics
    print("\n" + "=" * 65)
    print(f"    FINAL CROSS-VALIDATION SUMMARY ({num_completed_folds}-FOLD AVERAGE)")
    print("=" * 65)

    print("\n--- Validation Performance (Averaged across Folds) ---")
    for m in ["accuracy", "precision", "recall", "f1"]:
        mean_val = np.mean(val_metrics[m])
        std_val = np.std(val_metrics[m])
        print(f"{m.capitalize():<12}: {mean_val:.4f} ± {std_val:.4f}")

    print("\n--- Per-Fold Validation Breakdown ---")
    print(val_summary_df.to_string(index=False))

    print("\n" + "-" * 65)

    print("\n--- Holdout Test Performance (Averaged across Fold Models) ---")
    for m in ["accuracy", "precision", "recall", "f1"]:
        mean_test = np.mean(test_metrics[m])
        std_test = np.std(test_metrics[m])
        print(f"{m.capitalize():<12}: {mean_test:.4f} ± {std_test:.4f}")

    print("\n--- Per-Fold Holdout Test Breakdown ---")
    print(test_summary_df.to_string(index=False))
    print("=" * 65)

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_summary_df": val_summary_df,
        "test_summary_df": test_summary_df,
    }