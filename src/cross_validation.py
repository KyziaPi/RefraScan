# src/cross_validation.py
import os
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit

from src.preprocessing import (
    CLASS_NAMES,
    calculate_class_weights,
    create_image_generator,
    create_multimodal_generator,
    scale_age_feature,
)
from src.models import build_model, set_fine_tuning
from src.train import train_model
from src.evaluate import evaluate_model, metrics_to_row


RANDOM_STATE = 42

MODEL_INPUT_SHAPES = {
    "efficientnet": (300, 300, 3),
    "resnet50": (224, 224, 3),
    "densenet121": (224, 224, 3),
}


def get_input_image_shape(model_name):
    model_name = model_name.lower()

    if model_name not in MODEL_INPUT_SHAPES:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Choose from {list(MODEL_INPUT_SHAPES.keys())}."
        )

    return MODEL_INPUT_SHAPES[model_name]


def split_holdout_test(
    df,
    patient_col="ID",
    target_col="classification_encoded",
    test_size=0.15,
    random_state=42,
):
    """
    Create an untouched patient-level holdout set.

    The split is performed at the patient level so that both eyes
    belonging to the same patient remain in the same partition.

    Note:
    We intentionally do NOT assign one target label to each patient
    because a patient's two eyes may have different refractive-error
    classifications.
    """

    from sklearn.model_selection import GroupShuffleSplit

    # ---------------------------------------------------------------
    # Patient-level group split
    # ---------------------------------------------------------------
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_indices, holdout_indices = next(
        splitter.split(
            df,
            groups=df[patient_col]
        )
    )

    development_df = (
        df.iloc[train_indices]
        .copy()
        .reset_index(drop=True)
    )

    holdout_df = (
        df.iloc[holdout_indices]
        .copy()
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Verify there is absolutely no patient overlap
    # ---------------------------------------------------------------
    development_patients = set(
        development_df[patient_col]
    )

    holdout_patients = set(
        holdout_df[patient_col]
    )

    overlap = development_patients.intersection(
        holdout_patients
    )

    if overlap:
        raise ValueError(
            f"Patient leakage detected! "
            f"{len(overlap)} patients appear in both development "
            f"and holdout sets."
        )

    # ---------------------------------------------------------------
    # Report split information
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PATIENT-LEVEL HOLDOUT SPLIT")
    print("=" * 70)

    print(
        f"Development records : {len(development_df):,}"
    )

    print(
        f"Holdout records     : {len(holdout_df):,}"
    )

    print(
        f"Development patients: "
        f"{development_df[patient_col].nunique():,}"
    )

    print(
        f"Holdout patients    : "
        f"{holdout_df[patient_col].nunique():,}"
    )

    print(
        f"Patient overlap     : {len(overlap)}"
    )

    print(
        "\nThe holdout set is now untouched and will not be used "
        "for architecture/model selection."
    )

    return development_df, holdout_df


def _build_generators(
    train_df,
    val_df,
    preprocess_input,
    batch_size,
    target_size,
    use_metadata=False,
    metadata_cols=None,
):
    """Build training and validation generators using identical image rules."""
    if use_metadata:
        train_gen = create_multimodal_generator(
            train_df,
            metadata_cols=metadata_cols,
            batch_size=batch_size,
            target_size=target_size,
            augment=True,
            preprocess_fn=preprocess_input,
            shuffle=True,
        )
        val_gen = create_multimodal_generator(
            val_df,
            metadata_cols=metadata_cols,
            batch_size=batch_size,
            target_size=target_size,
            augment=False,
            preprocess_fn=preprocess_input,
            shuffle=False,
        )
    else:
        train_gen = create_image_generator(
            train_df,
            batch_size=batch_size,
            target_size=target_size,
            augment=True,
            preprocess_fn=preprocess_input,
            shuffle=True,
        )
        val_gen = create_image_generator(
            val_df,
            batch_size=batch_size,
            target_size=target_size,
            augment=False,
            preprocess_fn=preprocess_input,
            shuffle=False,
        )

    return train_gen, val_gen


def _build_test_generator(
    test_df,
    preprocess_input,
    batch_size,
    target_size,
    use_metadata=False,
    metadata_cols=None,
):
    """Create a deterministic holdout generator for the final evaluation only."""
    if use_metadata:
        return create_multimodal_generator(
            test_df,
            metadata_cols=metadata_cols,
            batch_size=batch_size,
            target_size=target_size,
            augment=False,
            preprocess_fn=preprocess_input,
            shuffle=False,
        )

    return create_image_generator(
        test_df,
        batch_size=batch_size,
        target_size=target_size,
        augment=False,
        preprocess_fn=preprocess_input,
        shuffle=False,
    )


def _print_summary(results_df, title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(results_df.to_string(index=False))

    for metric in ["Macro F1", "Balanced Accuracy", "Macro Precision", "Macro Recall", "Accuracy"]:
        print(
            f"{metric:<20}: "
            f"{results_df[metric].mean():.4f} ± {results_df[metric].std(ddof=0):.4f}"
        )


def run_cross_validation(
    df,
    model_name="efficientnet",
    preprocess_input=None,
    patient_col="ID",
    target_col="classification_encoded",
    n_splits=10,
    batch_size=16,
    epochs=30,
    learning_rate=1e-4,
    use_metadata=False,
    metadata_cols=None,
    output_dir="artifacts",
    fine_tune=False,
    fine_tune_layers=30,
    fine_tune_epochs=15,
    fine_tune_learning_rate=1e-5,
    fine_tune_stages=None,
):
    """
    Run the controlled development-data experiment.

    IMPORTANT: this function evaluates only validation folds. The 15% holdout
    is returned separately and is never used to select a model or hyperparameter.
    """
    if preprocess_input is None:
        raise ValueError("A backbone-specific preprocess_input function is required.")

    input_image_shape = get_input_image_shape(model_name)

    # Ensure experiment artifacts directory exists
    os.makedirs(output_dir, exist_ok=True)

    if use_metadata and metadata_cols is None:
        raise ValueError("metadata_cols must be provided when use_metadata=True.")

    # -------------------------------------------------------------------------
    # 1. Isolate the untouched 15% holdout BEFORE cross-validation.
    # -------------------------------------------------------------------------
    development_df, holdout_df = split_holdout_test(
        df,
        patient_col=patient_col,
        target_col=target_col,
    )

    # -------------------------------------------------------------------------
    # 2. Ten-fold Stratified Group CV on the remaining 85%.
    # -------------------------------------------------------------------------
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_rows = []
    fold_models = []

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} | {'IMAGE + AGE' if use_metadata else 'IMAGE ONLY'}")
    print(f"{n_splits}-FOLD STRATIFIED GROUP CROSS-VALIDATION")
    print("=" * 70)

    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(
            development_df,
            y=development_df[target_col],
            groups=development_df[patient_col],
        ),
        start=1,
    ):
        print(f"\n--- Fold {fold}/{n_splits} ---")

        train_df = development_df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = development_df.iloc[val_idx].copy().reset_index(drop=True)

        # Explicit leakage check for this fold.
        train_patients = set(train_df[patient_col])
        val_patients = set(val_df[patient_col])
        overlap = train_patients & val_patients
        if overlap:
            raise RuntimeError(f"Patient leakage detected in fold {fold}: {overlap}")

        # Age is fitted ONLY on the training fold when it is part of the ablation.
        if use_metadata:
            train_df, val_df = scale_age_feature(
                train_df,
                val_df,
                age_col="age",
                scaler_save_path=os.path.join(
                    output_dir, f"{model_name}_fold_{fold}_age_scaler.pkl"
                ),
            )

        # Class weights are calculated from the training fold only. We do not
        # simultaneously use the old class-balanced sampler; this keeps the
        # imbalance strategy controlled and reproducible.
        class_weights = calculate_class_weights(train_df, target_col)

        train_gen, val_gen = _build_generators(
            train_df,
            val_df,
            preprocess_input,
            batch_size,
            target_size=input_image_shape[:2],
            use_metadata=use_metadata,
            metadata_cols=metadata_cols,
        )

        train_steps = math.ceil(len(train_df) / batch_size)
        val_steps = math.ceil(len(val_df) / batch_size)

        model = build_model(
            model_name=model_name,
            input_image_shape=input_image_shape,
            num_metadata_features=len(metadata_cols or []),
            num_classes=3,
            dropout_rate=0.4,
            use_metadata=use_metadata,
        )

        frozen_path = os.path.join(
            output_dir, f"{model_name}_{'age' if use_metadata else 'image'}_fold_{fold}_frozen.weights.h5"
        )

        # ---------------------------------------------------------------------
        # 3. Train the frozen-backbone model first.
        # ---------------------------------------------------------------------
        train_model(
            model=model,
            train_ds=train_gen,
            val_ds=val_gen,
            epochs=epochs,
            learning_rate=learning_rate,
            steps_per_epoch=train_steps,
            validation_steps=val_steps,
            save_path=frozen_path,
            class_weight=class_weights,
        )

        model.load_weights(frozen_path)

        # ---------------------------------------------------------------------
        # 4. Optional controlled fine-tuning of upper backbone layers.
        # ---------------------------------------------------------------------
        if fine_tune:
            # Progressive fine-tuning: begin with a small number of upper
            # layers, then expand the trainable region while keeping the
            # learning rate low. The exact stages are fixed before training.
            stages = fine_tune_stages or [fine_tune_layers]
            saved_path = frozen_path

            for stage_number, trainable_layers in enumerate(stages, start=1):
                print(
                    f"\nFine-tuning stage {stage_number}: "
                    f"upper {trainable_layers} backbone layers..."
                )
                set_fine_tuning(
                    model, model_name, trainable_layers=trainable_layers
                )

                fine_path = os.path.join(
                    output_dir,
                    f"{model_name}_{'age' if use_metadata else 'image'}_fold_{fold}_stage_{stage_number}.weights.h5",
                )

                # Recreate the finite validation generator for each fine-tuning
                # stage because a finite generator is consumed by model.fit().
                _, stage_val_gen = _build_generators(
                    train_df,
                    val_df,
                    preprocess_input,
                    batch_size,
                    target_size=input_image_shape[:2],
                    use_metadata=use_metadata,
                    metadata_cols=metadata_cols,
                )

                train_model(
                    model=model,
                    train_ds=train_gen,
                    val_ds=stage_val_gen,
                    epochs=fine_tune_epochs,
                    learning_rate=fine_tune_learning_rate,
                    steps_per_epoch=train_steps,
                    validation_steps=val_steps,
                    save_path=fine_path,
                    class_weight=class_weights,
                )
                model.load_weights(fine_path)
                saved_path = fine_path
        else:
            saved_path = frozen_path

        # ---------------------------------------------------------------------
        # 5. Validation is the ONLY evaluation used during model selection.
        # ---------------------------------------------------------------------
        # Recreate validation generator so evaluation starts at the first
        # validation record rather than at the generator's previous state.
        eval_val_gen = (
            create_multimodal_generator(
                val_df,
                metadata_cols=metadata_cols,
                batch_size=batch_size,
                target_size=input_image_shape[:2],
                augment=False,
                preprocess_fn=preprocess_input,
                shuffle=False,
            )
            if use_metadata
            else create_image_generator(
                val_df,
                batch_size=batch_size,
                target_size=input_image_shape[:2],
                augment=False,
                preprocess_fn=preprocess_input,
                shuffle=False,
            )
        )

        val_result = evaluate_model(
            model=model,
            test_ds=eval_val_gen,
            y_true=val_df[target_col].values,
            steps=val_steps,
            class_names=CLASS_NAMES,
        )
        fold_rows.append(metrics_to_row(val_result, fold=fold, model_name=model_name))
        fold_models.append(saved_path)

    results_df = pd.DataFrame(fold_rows)
    _print_summary(
        results_df,
        f"{model_name.upper()} | {'IMAGE + AGE' if use_metadata else 'IMAGE ONLY'} CV SUMMARY",
    )

    return {
        "development_df": development_df,
        "holdout_df": holdout_df,
        "fold_results": results_df,
        "fold_model_paths": fold_models,
    }


def evaluate_holdout_once(
    model,
    holdout_df,
    preprocess_input,
    batch_size=16,
    use_metadata=False,
    metadata_cols=None,
    class_names=CLASS_NAMES,
):
    """
    Evaluate the untouched holdout exactly once.

    This function must only be called after architecture/model configuration has
    already been selected using development-data cross-validation.
    """
    holdout_gen = _build_test_generator(
        holdout_df,
        preprocess_input,
        batch_size,
        use_metadata=use_metadata,
        metadata_cols=metadata_cols,
    )
    holdout_steps = math.ceil(len(holdout_df) / batch_size)

    result = evaluate_model(
        model=model,
        test_ds=holdout_gen,
        y_true=holdout_df["classification_encoded"].values,
        steps=holdout_steps,
        class_names=class_names,
    )

    print("\n*** FINAL HOLDOUT EVALUATION COMPLETED ONCE ***")
    return result


def train_final_on_development(
    development_df,
    model_name,
    preprocess_input,
    batch_size=16,
    epochs=30,
    learning_rate=1e-4,
    use_metadata=False,
    metadata_cols=None,
    output_path="artifacts/final_model.weights.h5",
    fine_tune=False,
    fine_tune_layers=30,
    fine_tune_epochs=15,
    fine_tune_learning_rate=1e-5,
    fine_tune_stages=None,
):
    """
    Train the selected final configuration on all 85% development data.

    No holdout labels are used here. The resulting model is then ready for the
    single final holdout evaluation.
    """
    train_df = development_df.copy().reset_index(drop=True)
    input_image_shape = get_input_image_shape(model_name)

    if use_metadata:
        # For the final model, fit the age scaler on ALL development records.
        train_df, _ = scale_age_feature(
            train_df,
            train_df.copy(),
            age_col="age",
            scaler_save_path=os.path.join(
                os.path.dirname(output_path) or ".",
                "final_model_age_scaler.pkl",
            ),
        )
        metadata_cols = metadata_cols or ["age_scaled"]

    class_weights = calculate_class_weights(train_df, "classification_encoded")

    train_gen, _ = _build_generators(
        train_df,
        train_df,
        preprocess_input,
        batch_size,
        use_metadata=use_metadata,
        metadata_cols=metadata_cols,
    )

    model = build_model(
        model_name=model_name,
        input_image_shape=input_image_shape,
        num_metadata_features=len(metadata_cols or []),
        num_classes=3,
        dropout_rate=0.4,
        use_metadata=use_metadata,
    )

    steps = math.ceil(len(train_df) / batch_size)

    # We use a small validation subset only for early stopping during the final
    # fit. This subset is drawn from development data and never from holdout.
    # The model selection decision has already been made by CV at this point.
    history = train_model(
        model=model,
        train_ds=train_gen,
        val_ds=None,
        epochs=epochs,
        learning_rate=learning_rate,
        steps_per_epoch=steps,
        validation_steps=None,
        save_path=output_path,
        class_weight=class_weights,
    )

    model.load_weights(output_path)

    if fine_tune:
        # Apply the same progressive fine-tuning schedule selected during CV.
        stages = fine_tune_stages or [fine_tune_layers]
        for stage_number, trainable_layers in enumerate(stages, start=1):
            set_fine_tuning(
                model, model_name, trainable_layers=trainable_layers
            )
            fine_path = (
                os.path.splitext(output_path)[0]
                + f"_finetune_stage_{stage_number}.weights.h5"
            )
            train_model(
                model=model,
                train_ds=train_gen,
                val_ds=None,
                epochs=fine_tune_epochs,
                learning_rate=fine_tune_learning_rate,
                steps_per_epoch=steps,
                validation_steps=None,
                save_path=fine_path,
                class_weight=class_weights,
            )
            model.load_weights(fine_path)

    return model, history
