import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3, ResNet50, DenseNet121

def build_efficientnet(
    input_image_shape=(300, 300, 3), 
    num_metadata_features=1, 
    num_classes=3,
    dropout_rate=0.4
):
    # Image Branch
    image_input = layers.Input(shape=input_image_shape, name="image_input")
    base_model = EfficientNetB3(include_top=False, weights='imagenet', input_tensor=image_input)
    base_model.trainable = False
    
    x_img = layers.GlobalAveragePooling2D()(base_model.output)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x_img)
    x_img = layers.Dropout(0.5)(x_img)
    x_img = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x_img)
    x_img = layers.Dropout(0.3)(x_img)

    # Metadata Branch
    meta_input = layers.Input(shape=(num_metadata_features,), name="meta_input")
    x_meta = layers.Dense(64, activation='relu')(meta_input)
    x_meta = layers.Dropout(0.3)(x_meta)

    # Fusion
    merged = layers.Concatenate()([x_img, x_meta])
    x = layers.BatchNormalization()(merged)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(num_classes, activation='softmax', name="classification_output")(x)

    # Return uncompiled model
    return models.Model(inputs=[image_input, meta_input], outputs=output)

def build_resnet50(
    input_image_shape=(224, 224, 3), 
    num_metadata_features=1, 
    num_classes=3,
    dropout_rate=0.4
):
    # Image Branch
    image_input = layers.Input(shape=input_image_shape, name="image_input")
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=image_input)
    base_model.trainable = False

    base_model.trainable = False

    x_img = layers.GlobalAveragePooling2D()(base_model.output)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x_img)
    x_img = layers.Dropout(dropout_rate)(x_img)

    # Metadata Branch
    meta_input = layers.Input(shape=(num_metadata_features,), name="meta_input")
    x_meta = layers.Dense(64, activation='relu')(meta_input)
    x_meta = layers.Dropout(0.3)(x_meta)

    # Fusion
    merged = layers.Concatenate()([x_img, x_meta])
    x = layers.Dense(128, activation='relu')(merged)
    x = layers.Dropout(dropout_rate)(x)

    output = layers.Dense(num_classes, activation='softmax', name="classification_output")(x)

    return models.Model(inputs=[image_input, meta_input], outputs=output)

def build_densenet121(
    input_image_shape=(224, 224, 3), 
    num_metadata_features=1, 
    num_classes=3,
    dropout_rate=0.4,
):
    # Image Branch
    image_input = layers.Input(shape=input_image_shape, name="image_input")

    base_model = DenseNet121(
        include_top=False, 
        weights='imagenet', 
        input_tensor=image_input
    )

    # Freeze the pretrained backbone
    base_model.trainable = False


    x_img = layers.GlobalAveragePooling2D()(base_model.output)
    x_img = layers.BatchNormalization()(x_img)
    x_img = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x_img)
    x_img = layers.Dropout(0.5)(x_img)
    x_img = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x_img)
    x_img = layers.Dropout(0.3)(x_img)

    # Metadata Branch
    meta_input = layers.Input(shape=(num_metadata_features,), name="meta_input")
    x_meta = layers.Dense(64, activation='relu')(meta_input)
    x_meta = layers.Dropout(0.3)(x_meta)

    # Fusion
    merged = layers.Concatenate()([x_img, x_meta])

    x= layers.BatchNormalization()(merged)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.Dropout(0.3)(x)

    output = layers.Dense(
        num_classes,
        activation='softmax',
        name="classification_output"
    )(x)

    return models.Model(
        inputs=[image_input, meta_input],
        outputs=output
    )

def build_model(
    model_name,
    input_image_shape, 
    num_metadata_features=1, 
    num_classes=3,
    dropout_rate=0.4,
):
    if model_name == "efficientnet":
        return build_efficientnet(input_image_shape, num_metadata_features, num_classes, dropout_rate)
    elif model_name == "resnet50":
        return build_resnet50(input_image_shape, num_metadata_features, num_classes, dropout_rate)
    elif model_name == "densenet121":
        return build_densenet121(input_image_shape, num_metadata_features, num_classes, dropout_rate)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from 'efficientnet', 'resnet50', or 'densenet121'.")