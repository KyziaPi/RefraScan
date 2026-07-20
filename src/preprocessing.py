# src/preprocessing.py
import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import class_weight
import tensorflow as tf

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

def load_and_preprocess_image(img_path, target_size=(300, 300)):
    """Standardized image reader for custom generator pipelines."""
    try:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        return img
    except Exception as e:
        # Fallback to zeros if image fails to load during training
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

def create_multimodal_generator(df, numerical_cols, batch_size=16, target_size=(300, 300), preprocess_fn=None):
    """Custom generator yields [images, metadata] and targets."""
    num_samples = len(df)
    while True:
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        for offset in range(0, num_samples, batch_size):
            batch_df = df_shuffled.iloc[offset:offset+batch_size]
            
            images = []
            metadata = []
            labels = []
            
            for _, row in batch_df.iterrows():
                img = load_and_preprocess_image(row['full_path'], target_size)
                if preprocess_fn:
                    img = preprocess_fn(img)
                    
                images.append(img)
                metadata.append(row[numerical_cols].values.astype(np.float32))
                labels.append(row['Diagnosis_encoded']) # Adjust based on your target col name
                
            yield [np.array(images), np.array(metadata)], np.array(labels)