import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.preprocessing import (
    calculate_class_weights,
    create_multimodal_generator,
    scale_age_feature,
)
from src.models import build_model
from src.train import train_model
from src.evaluate import evaluate_model


def run_cross_validation(
    df: pd.DataFrame,
    model_name: str = "efficientnet",
    preprocess_input: callable = None,
    patient_col: str = "ID",
    target_col: str = "classification_encoded",
    n_splits: int = 10,
    batch_size: int = 16,
    epochs: int = 30,
):
    """Executes k-fold cross-validation using StratifiedGroupKFold."""
    
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    fold_metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    print(f"\n{'='*50}")
    print(f"STARTING {n_splits}-FOLD CROSS VALIDATION")
    print(f"{'='*50}\n")

    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(df, y=df[target_col], groups=df[patient_col])
    ):
        print(f"\n--- Fold {fold + 1}/{n_splits} ---")
        
        # 1. Split data for the current fold
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        # 2. Prevent Data Leakage: Scale age strictly using the fold's training set
        train_df, val_df = scale_age_feature(
            train_df,
            val_df,
            test_df=None,
            age_col="age",
            scaler_save_path=None  # Not saving the scaler for now
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
            preprocess_fn=preprocess_input
        )
        val_gen = create_multimodal_generator(
            val_df, 
            metadata_cols,
            class_weights=None,
            batch_size=batch_size, 
            augment=False, 
            preprocess_fn=preprocess_input
        )

        # 5. Build a fresh model for each fold
        model = build_model(
            model_name=model_name,
            input_image_shape=(300, 300, 3), 
            num_metadata_features=len(metadata_cols), 
            num_classes=3,
            dropout_rate=0.4
        )

        fold_model_path = f"models/best_model_fold_{fold + 1}.h5"

        # 6. Train the model
        train_model(
            model=model,
            train_ds=train_gen,
            val_ds=val_gen,
            epochs=epochs,
            learning_rate=0.0001,
            save_path=fold_model_path
        )

        # 7. Evaluate the best model on the fold's validation set
        model.load_weights(fold_model_path)
        metrics = evaluate_model(
            model=model, 
            test_ds=val_gen, 
            class_names=["Emmetropia", "Myopia", "Hyperopia"]
        )

        # 8. Store results
        fold_metrics["accuracy"].append(metrics.get("accuracy", 0))
        fold_metrics["precision"].append(metrics.get("precision", 0))
        fold_metrics["recall"].append(metrics.get("recall", 0))
        fold_metrics["f1"].append(metrics.get("f1", 0))

    # 9. Calculate and display final average metrics
    print(f"\n{'='*50}")
    print(f"CROSS VALIDATION RESULTS ({n_splits} Folds)")
    print(f"{'='*50}")
    
    print(f"Average Accuracy  : {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
    print(f"Average Precision : {np.mean(fold_metrics['precision']):.4f} ± {np.std(fold_metrics['precision']):.4f}")
    print(f"Average Recall    : {np.mean(fold_metrics['recall']):.4f} ± {np.std(fold_metrics['recall']):.4f}")
    print(f"Average F1 Score  : {np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}")
        
    return fold_metrics