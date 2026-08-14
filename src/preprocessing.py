import os
import random
import cv2
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

CLASS_NAMES = ["Emmetropia", "Myopia", "Hyperopia"]
CLASS_MAPPING = {"Emmetropia": 0, "Myopia": 1, "Hyperopia": 2}

# Refractive measurements are intentionally excluded from every model input.
TARGET_DERIVED_COLUMNS = {
    "sphere",
    "cylinder",
    "spherical_equivalent",
    "spherical equivalent",
    "spherical_equivalent_d",
    "se",
}

def load_and_clean_data(csv_path, img_dir):
    """Loads the dataset and maps filenames natively for Kaggle environment."""
    df = pd.read_csv(csv_path)
    required_cols = {"ID", "eye_side", "classification"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required dataset columns: {sorted(missing)}")
    
    def create_filename(row):
        side = "OD" if str(row['eye_side'].lower()) == 'right' else "OS"
        return f"RET{str(row['ID']).zfill(3)}{side}.jpg"
    
    df['full_path'] = df.apply(lambda r: os.path.join(img_dir, create_filename(r)), axis=1)
    return df

# Added a function to validate the dataset before training
def validate_dataset(df, patient_col="ID", target_col="classification"):
    """
    Perform the initial dataset checks required before model training.

    This function deliberately does not use refractive measurements as model
    inputs. If sphere/cylinder/spherical-equivalent columns are present, they
    are reported as target-derived fields and must not enter the generators.
    """
    print("\n" + "=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing = [c for c in [patient_col, target_col, "full_path"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Total images/records : {len(df)}")
    print(f"Unique patients      : {df[patient_col].nunique()}")
    print(f"Missing labels       : {df[target_col].isna().sum()}")
    print(f"Duplicate rows       : {df.duplicated().sum()}")

    if df[target_col].isna().any():
        raise ValueError("Some records have missing target labels.")

    invalid_labels = set(df[target_col].unique()) - set(CLASS_MAPPING.keys())
    if invalid_labels:
        raise ValueError(f"Unexpected classification labels: {invalid_labels}")

    print("\nClass distribution:")
    print(df[target_col].value_counts().reindex(CLASS_NAMES, fill_value=0))

    # Both eyes must remain in the same split. We explicitly inspect whether
    # a patient has more than one class before proceeding.
    patient_label_counts = df.groupby(patient_col)[target_col].nunique()
    mixed_patients = patient_label_counts[patient_label_counts > 1]
    print(f"\nPatients with multiple target classes: {len(mixed_patients)}")
    if len(mixed_patients) > 0:
        print(mixed_patients.head(10))
        raise ValueError(
            "Some patients have different target classes between records. "
            "Resolve these cases before patient-level stratified splitting."
        )

    # Explicitly identify target-derived measurements so they can never be
    # accidentally added to metadata_cols.
    derived_present = [c for c in df.columns if c.lower() in TARGET_DERIVED_COLUMNS]
    if derived_present:
        print("\nTarget-derived refractive columns detected and EXCLUDED from model inputs:")
        print(derived_present)

    # Verify image files before training.
    exists = df["full_path"].map(os.path.isfile)
    print(f"\nExisting image files : {exists.sum()}/{len(exists)}")
    if not exists.all():
        missing_paths = df.loc[~exists, "full_path"].head(10).tolist()
        print("Example missing paths:", missing_paths)
        raise FileNotFoundError("One or more referenced fundus images do not exist.")

    # Check that patients have both eyes when expected, but do not force this
    # as a requirement because the dataset may legitimately contain one eye.
    if "eye_side" in df.columns:
        eye_counts = df.groupby(patient_col)["eye_side"].nunique()
        print(f"Patients with >=2 eye records: {(eye_counts >= 2).sum()}/{len(eye_counts)}")

    return df

def encode_target(df, source_col="classification", target_col="classification_encoded"):
    """Encode the three clinical classes using the fixed class mapping."""
    df = df.copy()
    df[target_col] = df[source_col].map(CLASS_MAPPING)
    if df[target_col].isna().any():
        raise ValueError("Target encoding produced missing values.")
    df[target_col] = df[target_col].astype(int)
    return df

def majority_class_baseline(df, target_col="classification_encoded"):
    """Return the majority-class baseline for reference."""
    counts = df[target_col].value_counts().sort_index()
    majority_class = int(counts.idxmax())
    accuracy = float(counts.max() / counts.sum())
    return {
        "majority_class": majority_class,
        "majority_class_name": CLASS_NAMES[majority_class],
        "accuracy": accuracy,
        "class_counts": counts.to_dict(),
    }

def load_and_preprocess_image(img_path, target_size=(300, 300)):
    """Load an RGB fundus image and resize it with preserved aspect ratio."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = np.zeros((th, tw, 3), dtype=np.uint8)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    padded[top : top + nh, left : left + nw] = resized
    return padded

def augment_image(img):
    """Apply the same mild augmentation policy to every architecture."""
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]

    # Keep augmentation independent of the target label. This prevents the
    # augmentation policy itself from becoming a class-specific signal.
    if random.random() < 0.50:
        angle = random.uniform(-10.0, 10.0)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    if random.random() < 0.50:
        img = cv2.flip(img, 1)

    if random.random() < 0.30:
        brightness = random.uniform(-0.08, 0.08) * 255.0
        contrast = random.uniform(0.90, 1.10)
        img = (img - 127.5) * contrast + 127.5 + brightness

    if random.random() < 0.25:
        crop_factor = random.uniform(0.94, 0.98)
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        img = cv2.resize(
            img[top : top + new_h, left : left + new_w],
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )

    return np.clip(img, 0.0, 255.0).astype(np.float32)


def create_image_generator(
    df,
    batch_size=16,
    target_size=(300, 300),
    augment=False,
    preprocess_fn=None,
    shuffle=False,
):
    """
    Create a finite deterministic generator for validation/test and a shuffled
    generator for training.

    Unlike the original implementation, validation/test records are not sampled
    randomly with replacement. Every evaluation sample is seen exactly once.
    """
    df_copy = df.copy().reset_index(drop=True)
    indices = np.arange(len(df_copy))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            images = []
            labels = []

            for idx in batch_indices:
                row = df_copy.iloc[idx]
                img = load_and_preprocess_image(row["full_path"], target_size)
                if augment:
                    img = augment_image(img)
                if preprocess_fn is not None:
                    img = preprocess_fn(img)
                images.append(img)
                labels.append(int(row["classification_encoded"]))

            yield np.asarray(images, dtype=np.float32), np.asarray(labels, dtype=np.int32)

        if not shuffle:
            break


def create_multimodal_generator(
    df,
    metadata_cols,
    batch_size=16,
    target_size=(300, 300),
    augment=False,
    preprocess_fn=None,
    shuffle=False,
):
    """Finite generator for the final FUNDUS IMAGE + AGE ablation."""
    if not metadata_cols:
        raise ValueError("metadata_cols must contain at least one feature.")

    df_copy = df.copy().reset_index(drop=True)
    metadata = df_copy[metadata_cols].astype(np.float32).values
    indices = np.arange(len(df_copy))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            images = []
            meta_batch = []
            labels = []

            for idx in batch_indices:
                row = df_copy.iloc[idx]
                img = load_and_preprocess_image(row["full_path"], target_size)
                if augment:
                    img = augment_image(img)
                if preprocess_fn is not None:
                    img = preprocess_fn(img)
                images.append(img)
                meta_batch.append(metadata[idx])
                labels.append(int(row["classification_encoded"]))

            yield (
                np.asarray(images, dtype=np.float32),
                np.asarray(meta_batch, dtype=np.float32),
            ), np.asarray(labels, dtype=np.int32)

        if not shuffle:
            break


def calculate_class_weights(df, target_col):
    """Calculate weights from the training fold only."""
    classes = np.unique(df[target_col])
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=df[target_col].values,
    )
    return dict(zip(classes, weights))


def scale_age_feature(
    train_df,
    val_df,
    test_df=None,
    age_col="age",
    scaler_save_path=None,
):
    """Fit age scaling on training data only and apply it to other splits."""
    scaler = MinMaxScaler()
    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df[f"{age_col}_scaled"] = scaler.fit_transform(train_df[[age_col]])
    val_df[f"{age_col}_scaled"] = scaler.transform(val_df[[age_col]])

    if test_df is not None:
        test_df = test_df.copy()
        test_df[f"{age_col}_scaled"] = scaler.transform(test_df[[age_col]])

    if scaler_save_path:
        joblib.dump(scaler, scaler_save_path)

    if test_df is not None:
        return train_df, val_df, test_df
    return train_df, val_df
