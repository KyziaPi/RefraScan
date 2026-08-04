import numpy as np
import pandas as pd
import math
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.preprocessing import (
    calculate_class_weights,
    create_multimodal_generator,
    load_and_preprocess_image,
    scale_age_feature,
)
from src.models import build_model
from src.train import train_model, unfreeze_model
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
        f"Dataset Split: {len(cv_df)} samples for {10}-Fold CV | "
        f"{len(test_df)} samples in Holdout Test Set ({test_size*100:.0f}%)"
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
    epochs: int = 40,
    learning_rate: float = 3e-4,
    holdout_test_size: float = 0.15,

    fine_tune=True,
    fine_tune_epochs=20,
    fine_tune_lr=1e-5,
):

    cv_df, holdout_test_df = split_holdout_test(
        df,
        patient_col=patient_col,
        target_col=target_col,
        test_size=holdout_test_size,
    )

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    val_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    test_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    print(f"\n{'='*65}")
    print(f"STARTING {n_splits}-FOLD CROSS VALIDATION")
    print(f"{'='*65}\n")

    for fold, (train_idx, val_idx) in enumerate(

        sgkf.split(
            cv_df,
            y=cv_df[target_col],
            groups=cv_df[patient_col],
        )

    ):

        print(f"\n--- Fold {fold+1}/{n_splits} ---")

        train_df = cv_df.iloc[train_idx].copy()
        val_df = cv_df.iloc[val_idx].copy()

        train_df, val_df, current_test_df = scale_age_feature(
            train_df=train_df,
            val_df=val_df,
            test_df=holdout_test_df.copy(),
            age_col="age",
            scaler_save_path=f"scaler_fold_{fold+1}.pkl",
        )

        metadata_cols = ["age_scaled"]

        class_weights = calculate_class_weights(
            train_df,
            target_col,
        )

        train_gen = create_multimodal_generator(
            train_df,
            metadata_cols,
            class_weights,
            batch_size=batch_size,
            augment=True,
            preprocess_fn=preprocess_input,
            image_loader=load_and_preprocess_image,
        )

        val_gen = create_multimodal_generator(
            val_df,
            metadata_cols,
            class_weights=None,
            batch_size=batch_size,
            augment=False,
            preprocess_fn=preprocess_input,
            image_loader=load_and_preprocess_image,
        )

        test_gen = create_multimodal_generator(
            current_test_df,
            metadata_cols,
            class_weights=None,
            batch_size=batch_size,
            augment=False,
            preprocess_fn=preprocess_input,
            image_loader=load_and_preprocess_image,
        )

        majority_class_count = (
            train_df[target_col]
            .value_counts()
            .max()
        )

        effective_train_size = majority_class_count * 3

        steps_per_epoch = int(
            np.ceil(
                effective_train_size / batch_size
            )
        )

        validation_steps = math.ceil(
            len(val_df) / batch_size
        )

        test_steps = math.ceil(
            len(current_test_df) / batch_size
        )

        model = build_model(
            model_name=model_name,
            input_image_shape=(300, 300, 3),
            num_metadata_features=len(metadata_cols),
            num_classes=3,
            dropout_rate=0.6,
        )

        fold_model_path = (
            f"best_{model_name}_fold_{fold+1}.h5"
        )

        # -----------------------------
        # Initial Training
        # -----------------------------
        train_model(
            model=model,
            train_ds=train_gen,
            val_ds=val_gen,
            epochs=epochs,
            learning_rate=learning_rate,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            save_path=fold_model_path,
            class_weight=class_weights,
            patience=10,
        )

        model.load_weights(fold_model_path)

        # ==========================================
        # Stage 2 - Fine-tuning (optional)
        # ==========================================

        if fine_tune:

            model = unfreeze_model(model, model_name)

            fine_model_path = f"finetuned{model_name}_fold_{fold+1}.h5"

            train_model(
                model=model,
                train_ds=train_gen,
                val_ds=val_gen,
                epochs=fine_tune_epochs,
                learning_rate=fine_tune_lr,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                save_path=fine_model_path,
                class_weight=class_weights,
                patience=5,
            )

            model.load_weights(fine_model_path)

        print("\n[Validation Set Evaluation]")

        v_res = evaluate_model(
                model=model,
                test_ds=val_gen,
                steps=validation_steps,
                y_true=val_df[target_col].values,
                class_names=[
                    "Emmetropia",
                    "Myopia",
                    "Hyperopia",
                ],
            )

        print("\n[Holdout Test Set Evaluation]")

        t_res = evaluate_model(
            model=model,
            test_ds=test_gen,
            steps=test_steps,
            y_true=current_test_df[target_col].values,
            class_names=[
                "Emmetropia",
                "Myopia",
                "Hyperopia",
            ],
        )

        for metric in ["accuracy", "precision", "recall", "f1"]:

            val_metrics[metric].append(
                v_res.get(metric, 0)
            )

            test_metrics[metric].append(
                t_res.get(metric, 0)
            )

    val_summary_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(n_splits)],
        "Accuracy": val_metrics["accuracy"],
        "Precision": val_metrics["precision"],
        "Recall": val_metrics["recall"],
        "F1 Score": val_metrics["f1"],
    })

    test_summary_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(n_splits)],
        "Accuracy": test_metrics["accuracy"],
        "Precision": test_metrics["precision"],
        "Recall": test_metrics["recall"],
        "F1 Score": test_metrics["f1"],
    })

    print("\n" + "="*65)
    print("FINAL CROSS VALIDATION SUMMARY")
    print("="*65)

    print("\nValidation")

    for metric in val_metrics:

        print(
            f"{metric}: "
            f"{np.mean(val_metrics[metric]):.4f}"
            f" ± "
            f"{np.std(val_metrics[metric]):.4f}"
        )

    print("\nHoldout Test")

    for metric in test_metrics:

        print(
            f"{metric}: "
            f"{np.mean(test_metrics[metric]):.4f}"
            f" ± "
            f"{np.std(test_metrics[metric]):.4f}"
        )

    return {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_summary_df": val_summary_df,
        "test_summary_df": test_summary_df,
    }