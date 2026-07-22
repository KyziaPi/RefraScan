# src/preprocessing.py
import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
#from tensorflow.keras.applications.efficientnet import preprocess_input

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

def apply_clahe(img):
    """Applies CLAHE enhancement in LAB color space to boost retinal contrast."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
    return enhanced

def load_and_preprocess_image(img_path, data_origin, target_size=(300, 300)):
    """
    Loads image, applies CLAHE to ALL sources, and handles padding vs standard resize.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Apply CLAHE contrast enhancement on all images
        img = apply_clahe(img)
        
        # 2. Origin-specific spatial resize logic
        if data_origin == 'PAPILA':
            img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
            padded_tensor = tf.image.resize_with_pad(img_tensor, target_size[0], target_size[1])
            img = padded_tensor.numpy().astype(np.uint8)
        else:
            # AUFMC - Lasik and default fallback
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            
        return img
    except Exception as e:
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
def augment_image(img_tensor, label_code):
    """
    Applies image augmentations.
    Label codes: 0 = Emmetropia, 1 = Myopia, 2 = Hyperopia.
    Emmetropia (0) and Hyperopia (2) get extra heavy augmentations.
    """
    # Standard Augmentations (All Classes)
    img_tensor = tf.image.random_flip_left_right(img_tensor)
    img_tensor = tf.image.random_flip_up_down(img_tensor)
    img_tensor = tf.image.random_brightness(img_tensor, max_delta=0.15)
    img_tensor = tf.image.random_contrast(img_tensor, lower=0.85, upper=1.15)
    
    # Heavy Augmentations specifically for minority/target classes (Emmetropia & Hyperopia)
    if label_code in [0, 2]:
        # Random rotation (90-degree increments or slight shifts)
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        img_tensor = tf.image.rot90(img_tensor, k=k)
        
        # Random hue variation
        img_tensor = tf.image.random_hue(img_tensor, max_delta=0.05)
        
        # Random saturation shift
        img_tensor = tf.image.random_saturation(img_tensor, lower=0.8, upper=1.2)
        
    return tf.clip_by_value(img_tensor, 0.0, 255.0)

def create_multimodal_generator(df, metadata_cols, batch_size=16, target_size=(300, 300), augment=False, preprocess_fn=None):
    """
    Custom generator yields [images, metadata] and classification targets.
    Supports class-aware image augmentations for training.
    """
    num_samples = len(df)
    while True:
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        for offset in range(0, num_samples, batch_size):
            batch_df = df_shuffled.iloc[offset:offset+batch_size]
            
            images = []
            metadata = []
            labels = []
            
            for _, row in batch_df.iterrows():
                origin = row['data_origin']
                label_code = row['classification_encoded']
                
                # Load and preprocess base image
                img = load_and_preprocess_image(row['full_path'], data_origin=origin, target_size=target_size)
                img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
                
                # Apply class-specific augmentations if enabled
                if augment:
                    img_tensor = augment_image(img_tensor, label_code)
                
                # EfficientNet normalizations
                if preprocess_fn:
                    img_processed = preprocess_fn(img_tensor.numpy())
                else:
                    img_processed = img_tensor.numpy()
                    
                images.append(img_processed)
                metadata.append(row[metadata_cols].values.astype(np.float32))
                labels.append(label_code)
                
            yield [np.array(images), np.array(metadata)], np.array(labels)