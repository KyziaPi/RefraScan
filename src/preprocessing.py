# src/preprocessing.py
import os
import pandas as pd
import numpy as np
import cv2
import random
import tensorflow as tf
import joblib
from sklearn.preprocessing import MinMaxScaler
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

def preprocess_metadata(df, numerical_cols):
    """Encodes and scales tabular clinical features."""
    df = df.copy()
    
    # Scale numerical metadata
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols].fillna(df[numerical_cols].median()))
        
    return df, scaler

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
    
def augment_image(img_tensor, class_weight=1.0, label_code=None):
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
    # Ensure tensor is float32 to prevent clip_by_value uint8 errors
    img_tensor = tf.cast(img_tensor, tf.float32)

    # 1. Determine maximum allowed transformations based on label_code
    # (Mapping: 0=Emmetropia, 1=Myopia, 2=Hyperopia)
    if label_code == 1:       # Myopia
        max_transforms = random.choice([0, 1])
    elif label_code == 2:     # Hyperopia
        max_transforms = random.choice([0, 1, 2])
    elif label_code == 0:     # Emmetropia
        max_transforms = random.choice([1, 2, 3])
    else:                     # Default fallback
        max_transforms = 1

    if max_transforms == 0:
        return tf.clip_by_value(img_tensor, 0.0, 255.0)

    # 2. Define pool of mild transformation functions
    # Intensity factor scales linearly with class_weight (clamped between 0.5 and 1.5)
    intensity = max(0.5, min(float(class_weight), 1.5))

    def apply_rotation(img):
        # Angle ranges between ±(8° * intensity) up to ±15° max
        angle_deg = random.uniform(-10.0, 10.0) * intensity
        angle_rad = angle_deg * (3.14159265 / 180.0)
        # Using tfa/tf.image rotation or standard tf rotation fill
        return tf.image.stateless_random_brightness(img, max_delta=0.0, seed=(1, 2)) # structural placeholder/pure spatial fallback if needed

    def apply_flip(img):
        return tf.image.flip_left_right(img)

    def apply_brightness_contrast(img):
        # Small delta scaled by class weight
        max_delta = 0.08 * intensity
        img = tf.image.random_brightness(img, max_delta=max_delta)
        img = tf.image.random_contrast(img, lower=1.0 - (0.1 * intensity), upper=1.0 + (0.1 * intensity))
        return img

    def apply_zoom(img):
        # Small central crop and resize back to original dimensions (1-5% zoom)
        crop_factor = random.uniform(0.92, 0.98)
        shape = tf.shape(img)
        h, w = shape[0], shape[1]
        new_h, new_w = tf.cast(tf.cast(h, tf.float32) * crop_factor, tf.int32), tf.cast(tf.cast(w, tf.float32) * crop_factor, tf.int32)
        
        cropped = tf.image.random_crop(img, size=[new_h, new_w, shape[2]])
        return tf.image.resize(cropped, [h, w])

    def apply_translation(img):
        # Small spatial shift (up to ~5% max)
        pad_size = int(8 * intensity)
        shape = tf.shape(img)
        padded = tf.image.pad_to_bounding_box(img, pad_size, pad_size, shape[0] + pad_size * 2, shape[1] + pad_size * 2)
        cropped = tf.image.random_crop(padded, size=shape)
        return cropped

    # 3. Randomly select exact transforms up to max_transforms cap
    transform_pool = [apply_rotation, apply_flip, apply_brightness_contrast, apply_zoom, apply_translation]
    selected_transforms = random.sample(transform_pool, min(max_transforms, len(transform_pool)))

    # 4. Sequential execution of selected capped transformations
    for transform_fn in selected_transforms:
        img_tensor = transform_fn(img_tensor)

    return tf.clip_by_value(img_tensor, 0.0, 255.0)
    
def create_multimodal_generator(df, metadata_cols, class_weights, batch_size=16, target_size=(300, 300), augment=False, preprocess_fn=None):
    """
    Generator yielding multi-modal inputs: (images, metadata) and targets.
    Handles numeric columns (e.g., 'age') and converts categorical text columns
    (e.g., 'data_origin') via one-hot encoding into a numerical matrix.
    """
    df = df.copy()
    
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

    while True:
        # Shuffle indices each epoch
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_df = df.iloc[batch_indices]
            batch_meta = metadata_matrix[batch_indices]

            images = []
            targets = []

            for idx, (_, row) in enumerate(batch_df.iterrows()):
                # Load image
                img = load_and_preprocess_image(row['full_path'], target_size=target_size)
                
                # Apply class-aware / conditional augmentations if enabled
                if augment:
                    label = row.get('classification_encoded', None)
                    
                    # Retrieve class weight from class_weights
                    weight = class_weights.get(label)
                    
                    img = augment_image(img, class_weight=weight, label_code=label)

                # Apply model-specific preprocessing (e.g., EfficientNet preprocess_input)
                if preprocess_fn is not None:
                    img = preprocess_fn(img)

                images.append(img)

                if 'classification_encoded' in row:
                    targets.append(row['classification_encoded'])

            batch_images = np.array(images, dtype=np.float32)
            batch_targets = np.array(targets, dtype=np.int32) if targets else None

            if batch_targets is not None:
                yield [batch_images, batch_meta], batch_targets
            else:
                yield [batch_images, batch_meta]

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