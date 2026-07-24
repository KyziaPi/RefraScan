# src/preprocessing.py
import os
import pandas as pd
import numpy as np
import cv2
import random
import tensorflow as tf
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

def load_and_clean_data(csv_path, img_dir):
    """Loads the dataset and maps filenames natively for Kaggle environment."""
    df = pd.read_csv(csv_path)
    
    def create_filename(row):
        side = "OD" if row['eye_side'] == 'right' else "OS"
        return f"RET{str(row['ID']).zfill(3)}{side}.jpg"
    
    df['full_path'] = df.apply(lambda r: os.path.join(img_dir, create_filename(r)), axis=1)
    return df

def preprocess_metadata(df, numerical_cols, categorical_cols):
    """Encodes and scales tabular clinical features."""
    df = df.copy()
    
    # Scale numerical metadata
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols].fillna(df[numerical_cols].median()))
    
    # Encode categorical metadata
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    return df, scaler, encoders

# For future testing
def apply_clahe(img):
    """Applies CLAHE enhancement in LAB color space to boost retinal contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    return enhanced

def load_and_preprocess_image(img_path, target_size=(300, 300)):
    """
    Loads image, and handles padding vs standard resize.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Apply CLAHE contrast enhancement on all images
        #img = apply_clahe(img)
        
        # 2. Resize image
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        padded_tensor = tf.image.resize_with_pad(img_tensor, target_size[0], target_size[1])
        img = padded_tensor.numpy().astype(np.uint8)
            
        return img
    except Exception as e:
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
def augment_image(img, class_weight=1.0, label_code=None):
    """
    Applies capped number of light augmentations with intensity scaled by class_weight.
    
    Caps per original image:
      - Myopia (1): Max 1 transformation (or none)
      - Hyperopia (2): Max 2 transformations
      - Emmetropia (0): Max 3 transformations
      
    Allowed transforms (all mild):
      - Small rotation (±10° to ±15°)
      - Horizontal flip
      - Slight brightness/contrast adjustment
      - Small zoom (crop & resize)
      - Small translation (shift)
    """
    # Ensure tensor/array is converted to float32 NumPy array for processing
    img_np = np.array(img, dtype=np.float32)
    
    # 1. Determine maximum allowed transformations based on label_code
    # (Mapping: 0=Emmetropia, 1=Myopia, 2=Hyperopia)
    if label_code == 1:       # Myopia
        max_transforms = random.choice([0, 1])
    elif label_code == 2:     # Hyperopia
        max_transforms = random.choice([0, 1, 2])
    elif label_code == 0:     # Emmetropia
        max_transforms = random.choice([2, 3, 4])
    else:                     # Default fallback
        max_transforms = 1

    if max_transforms == 0:
        return np.clip(img_np, 0.0, 255.0).astype(np.float32)

    # 2. Intensity Factor: Allow higher intensity cap specifically for Emmetropia
    if label_code == 0:
        # Scale higher for Emmetropia (clamped between 1.0 and 2.0)
        intensity = max(1.0, min(float(class_weight), 2.0))
    else:
        intensity = max(0.5, min(float(class_weight), 1.5))

    h, w = img_np.shape[:2]

    def apply_rotation(img_arr):
        # Angle ranges between ±(8° * intensity) up to ±15° max
        angle_deg = random.uniform(-10.0, 10.0) * intensity
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        return cv2.warpAffine(img_arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def apply_flip(img_arr):
        return cv2.flip(img_arr, 1)

    def apply_brightness_contrast(img_arr):
        # Small delta scaled by class weight
        brightness_delta = random.uniform(-0.08, 0.08) * intensity * 255.0
        contrast_factor = random.uniform(
            1.0 - (0.1 * intensity), 1.0 + (0.1 * intensity)
        )
        adjusted = (
            (img_arr - 127.5) * contrast_factor + 127.5 + brightness_delta
        )
        return adjusted

    def apply_zoom(img_arr):
        # Small central crop and resize back to original dimensions (1-5% zoom)
        crop_factor = random.uniform(0.92, 0.98)
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        cropped = img_arr[top : top + new_h, left : left + new_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def apply_translation(img_arr):
        max_shift = max(2, int(12 * intensity))
        tx = random.randint(-max_shift, max_shift)
        ty = random.randint(-max_shift, max_shift)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        return cv2.warpAffine(img_arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def apply_shear(img_arr):
        shear_val = random.uniform(-0.08, 0.08) * intensity
        M = np.float32([[1, shear_val, 0], [0, 1, 0]])
        return cv2.warpAffine(img_arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 3. Randomly select exact transforms up to max_transforms cap
    transform_pool = [apply_rotation, apply_flip, apply_brightness_contrast, apply_zoom, apply_translation, apply_shear]
    selected_transforms = random.sample(transform_pool, min(max_transforms, len(transform_pool)))

    # 4. Sequential execution of selected capped transformations
    for transform_fn in selected_transforms:
        img_np = transform_fn(img_np)

    return np.clip(img_np, 0.0, 255.0).astype(np.float32)
    
def create_multimodal_generator(df, metadata_cols, class_weights, batch_size=16, target_size=(300, 300), augment=False, preprocess_fn=None):
    """
    Generator yielding multi-modal inputs: (images, metadata) and targets.
    Handles numeric columns (e.g., 'age')
    """
    df_copy = df.copy().reset_index(drop=True)
    
    # Pre-process metadata columns: handle categorical string variables via One-Hot Encoding
    processed_meta = []
    for col in metadata_cols:
        if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.CategoricalDtype):
            # One-hot encode string/categorical columns (e.g., data_origin)
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            processed_meta.append(dummies)
        else:
            # Numeric columns (e.g., age)
            processed_meta.append(df[[col]])
            
    # Concatenate processed metadata into a single DataFrame and convert safely to float32
    metadata_df = pd.concat(processed_meta, axis=1)
    metadata_matrix = metadata_df.values.astype(np.float32)

    num_samples = len(df)
    
    # Group indices by class for balanced sampling
    class_indices = {
        c: df_copy[df_copy["classification_encoded"] == c].index.tolist()
        for c in df_copy["classification_encoded"].unique()
    }

    while True:
        images = []
        metadata = []
        targets = []

        if augment:
            # --- TRAINING MODE: Equal Class-Balanced Sampling ---
            samples_per_class = batch_size // len(class_indices)
            selected_indices = []

            for c, idxs in class_indices.items():
                # Oversample minority classes with replacement
                selected_indices.extend(
                    np.random.choice(idxs, size=samples_per_class, replace=True)
                )

            # Fill any remainder slots to match exact batch_size
            remaining = batch_size - len(selected_indices)
            if remaining > 0:
                selected_indices.extend(
                    np.random.choice(df_copy.index, size=remaining, replace=True)
                )

            np.random.shuffle(selected_indices)
        else:
            # --- VAL / TEST MODE: Sequential / Standard Sampling ---
            selected_indices = np.random.choice(
                df_copy.index, size=batch_size, replace=False
            )

        for idx in selected_indices:
            row = df_copy.iloc[idx]

            # 1. Load image
            img_path = row["full_path"]
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (300, 300))

            # 2. Extract label & apply balanced augmentation
            label = row.get("classification_encoded", None)
            if augment:
                weight = class_weights.get(label, 1.0) if class_weights else 1.0
                img = augment_image(img, class_weight=weight, label_code=label)

            # 3. Preprocess image
            if preprocess_fn:
                img = preprocess_fn(img)

            images.append(img)

            # 4. Extract metadata
            meta_feat = row[metadata_cols].values.astype(np.float32)
            metadata.append(meta_feat)

            if label is not None:
                targets.append(label)

        # Convert to arrays and yield
        batch_images = np.array(images, dtype=np.float32)
        batch_meta = np.array(metadata, dtype=np.float32)
        batch_targets = np.array(targets, dtype=np.int32) if targets else None

        if batch_targets is not None:
            yield (batch_images, batch_meta), batch_targets
        else:
            yield (batch_images, batch_meta)

def split_data_by_patient(df, patient_col='ID', target_col='classification_encoded', test_size=0.30, val_ratio=0.50, random_state=42):
    """
    Splits a DataFrame by patient ID using stratified sampling to prevent data leakage 
    across bilateral eye samples while maintaining class balance.

    Parameters:
      df (pd.DataFrame): The input DataFrame.
      patient_col (str): Column name containing patient IDs.
      target_col (str): Column name containing target classification labels.
      test_size (float): Fraction of patients for temp split (Val + Test). Default 0.30 (70% Train).
      val_ratio (float): Fraction of temp split assigned to Val vs Test. Default 0.50 (15% Val, 15% Test).
      random_state (int): Seed for reproducibility.

    Returns:
      train_df, val_df, test_df (tuple of pd.DataFrame): DataFrames for train, val, and test splits.
    """
    df = df.copy()

    # Get primary classification label per unique patient ID for stratification
    patient_classes = df.groupby(patient_col)[target_col].first()
    unique_ids = patient_classes.index.values
    unique_labels = patient_classes.values

    # 1. First split: Train vs Temp (Val + Test)
    train_ids, temp_ids, _, temp_labels = train_test_split(
        unique_ids,
        unique_labels,
        test_size=test_size,
        stratify=unique_labels,
        random_state=random_state
    )

    # 2. Second split: Temp into Val and Test
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=val_ratio,
        stratify=temp_labels,
        random_state=random_state
    )

    # 3. Filter full dataset by patient IDs to preserve all eye images without leakage
    train_df = df[df[patient_col].isin(train_ids)].copy().reset_index(drop=True)
    val_df = df[df[patient_col].isin(val_ids)].copy().reset_index(drop=True)
    test_df = df[df[patient_col].isin(test_ids)].copy().reset_index(drop=True)

    return train_df, val_df, test_df

def calculate_class_weights(df, target_col):
    """Helper to balance gradients against clinical minority classes."""
    classes = np.unique(df[target_col])
    weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                 classes=classes, 
                                                 y=df[target_col].values)
    return dict(zip(classes, weights))

def scale_age_feature(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame = None,
    age_col: str = "age",
    scaler_save_path: str = "age_scaler.pkl",
):
  """Fits MinMaxScaler on train_df[age_col], transforms val_df and test_df,

  and appends an '{age_col}_scaled' column to each.
  """
  scaler = MinMaxScaler()

  # Create copies to prevent SettingWithCopy warnings
  train_df = train_df.copy()
  val_df = val_df.copy()

  # 1. Fit & transform on training data
  train_df[f"{age_col}_scaled"] = scaler.fit_transform(train_df[[age_col]])

  # 2. Transform validation data using training bounds
  val_df[f"{age_col}_scaled"] = scaler.transform(val_df[[age_col]])

  # 3. Transform test data if available
  if test_df is not None:
    test_df = test_df.copy()
    test_df[f"{age_col}_scaled"] = scaler.transform(test_df[[age_col]])

  # 4. Save fitted scaler artifact
  if scaler_save_path:
    joblib.dump(scaler, scaler_save_path)

  if test_df is not None:
    return train_df, val_df, test_df
  return train_df, val_df