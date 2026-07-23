# src/models.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3

def build_efficientnet(
    input_image_shape=(300, 300, 3), 
    num_metadata_features=5, 
    num_classes=3,
    learning_rate=0.0001,
    dropout_rate=0.4
):
    """
    Builds a multimodal model combining an EfficientNet image branch 
    and a dense metadata branch.
    """
    # -------------------------------------------------------------
    # Branch 1: Image Input & Feature Extraction
    # -------------------------------------------------------------
    image_input = layers.Input(shape=input_image_shape, name="image_input")
    
    # Load base model (weights='imagenet' for transfer learning)
    base_model = EfficientNetB3(
        include_top=False, 
        weights='imagenet', 
        input_tensor=image_input
    )
    
    # Freeze base model layers initially (optional, but recommended for stable early training)
    base_model.trainable = False 
    
    # Pool features from the image base
    x_img = layers.GlobalAveragePooling2D()(base_model.output)
    x_img = layers.Dense(256, activation='relu')(x_img)
    x_img = layers.Dropout(dropout_rate)(x_img)  # Applying dropout constraint

    # -------------------------------------------------------------
    # Branch 2: Metadata Input & Processing
    # -------------------------------------------------------------
    meta_input = layers.Input(shape=(num_metadata_features,), name="meta_input")
    x_meta = layers.Dense(64, activation='relu')(meta_input)
    x_meta = layers.Dropout(0.3)(x_meta) # Lower bound of the 0.3 - 0.5 range

    # -------------------------------------------------------------
    # Fusion & Output
    # -------------------------------------------------------------
    merged = layers.Concatenate()([x_img, x_meta])
    
    # Fully connected layers with dropout to prevent overfitting
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output layer (3 classes: Emmetropia, Myopia, Hyperopia)
    output = layers.Dense(num_classes, activation='softmax', name="classification_output")(x)

    # -------------------------------------------------------------
    # Compilation
    # -------------------------------------------------------------
    model = models.Model(inputs=[image_input, meta_input], outputs=output)
    
    # Adam Optimizer with initialized LR of 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy', # Used since labels are integers (0, 1, 2)
        metrics=['accuracy']
    )
    
    return model

def build_resnet50(
    input_image_shape=(224, 224, 3), 
    num_metadata_features=5, 
    num_classes=3,
    learning_rate=0.0001,
    dropout_rate=0.4
):
    """
    Stub for ResNet50 multimodal model.
    To be implemented.
    """
    pass


def build_densenet121(
    input_image_shape=(224, 224, 3), 
    num_metadata_features=5, 
    num_classes=3,
    learning_rate=0.0001,
    dropout_rate=0.4
):
    """
    Stub for DenseNet121 multimodal model.
    To be implemented.
    """
    pass