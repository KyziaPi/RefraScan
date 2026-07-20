# src/train.py
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications.efficientnet import EfficientNetB3
from sklearn.utils import class_weight

def build_efficientnet_multimodal(num_metadata_features, lr=1e-4):
    """Builds the late-fusion multi-modal EfficientNetB3 model architecture."""
    # Image Sub-network
    image_input = layers.Input(shape=(300, 300, 3), name='image_input')
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_tensor=image_input)
    base_model.trainable = True # Or False depending on your fine-tuning phase
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dropout(0.4)(x)
    
    # Metadata Sub-network
    meta_input = layers.Input(shape=(num_metadata_features,), name='metadata_input')
    y = layers.Dense(32, activation='relu')(meta_input)
    y = layers.BatchNormalization()(y)
    
    # Late Fusion
    combined = layers.concatenate([x, y])
    z = layers.Dense(128, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    
    # Assuming binary classification (e.g., Glaucoma vs Normal)
    output = layers.Dense(1, activation='sigmoid', name='output')(z)
    
    model = models.Model(inputs=[image_input, meta_input], outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def calculate_class_weights(df, target_col):
    """Helper to balance gradients against clinical minority classes."""
    classes = np.unique(df[target_col])
    weights = class_weight.compute_class_weight(class_weight='balanced', 
                                                 classes=classes, 
                                                 y=df[target_col].values)
    return dict(zip(classes, weights))