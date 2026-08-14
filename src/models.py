import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3, ResNet50, DenseNet121

# Backbone builders
# The IMAGE-ONLY experiments use the same input resolution (300x300) so that
# preprocessing and training conditions are controlled across architectures.
# Age is intentionally NOT included in these builders unless use_metadata=True.


def _build_image_branch(base_model, dropout_rate=0.4):
    """Create the common image feature head used by all three CNNs."""
    x_img = layers.GlobalAveragePooling2D(name="global_average_pooling")(base_model.output)
    x_img = layers.Dense(256, activation="relu", name="image_dense")(x_img)
    x_img = layers.Dropout(dropout_rate, name="image_dropout")(x_img)
    return x_img


def build_efficientnet(
    input_image_shape=(300, 300, 3), 
    num_metadata_features=0, 
    num_classes=3,
    dropout_rate=0.4,
    use_metadata=False,
    weights="imagenet"
):
    image_input = layers.Input(shape=input_image_shape, name="image_input")
    base_model = EfficientNetB3(
        include_top=False,
        weights=weights,
        input_tensor=image_input
    )
    base_model.trainable = False

    x_img = _build_image_branch(base_model, dropout_rate)

    if use_metadata:
        meta_input = layers.Input(shape=(num_metadata_features,), name="meta_input")
        x_meta = layers.Dense(64, activation="relu", name="metadata_dense")(meta_input)
        x_meta = layers.Dropout(0.3, name="metadata_dropout")(x_meta)
        merged = layers.Concatenate(name="feature_fusion")([x_img, x_meta])
        x = layers.Dense(128, activation="relu", name="fusion_dense")(merged)
        x = layers.Dropout(dropout_rate, name="fusion_dropout")(x)
        inputs = [image_input, meta_input]
    else:
        x = layers.Dense(128, activation="relu", name="classifier_dense")(x_img)
        x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
        inputs = image_input

    output = layers.Dense(num_classes, activation="softmax", name="classification_output")(x)
    return models.Model(inputs=inputs, outputs=output, name="efficientnetb3")

def build_resnet50(
    input_image_shape=(224, 224, 3), 
    num_metadata_features=0, 
    num_classes=3,
    dropout_rate=0.4,
    use_metadata=False,
):
    image_input = layers.Input(shape=input_image_shape, name="image_input")
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
        name="resnet50_backbone",
    )
    base_model.trainable = False

    x_img = _build_image_branch(base_model, dropout_rate)

    if use_metadata:
        meta_input = layers.Input(shape=(num_metadata_features,), name="meta_input")
        x_meta = layers.Dense(64, activation="relu", name="metadata_dense")(meta_input)
        x_meta = layers.Dropout(0.3, name="metadata_dropout")(x_meta)
        merged = layers.Concatenate(name="feature_fusion")([x_img, x_meta])
        x = layers.Dense(128, activation="relu", name="fusion_dense")(merged)
        x = layers.Dropout(dropout_rate, name="fusion_dropout")(x)
        inputs = [image_input, meta_input]
    else:
        x = layers.Dense(128, activation="relu", name="classifier_dense")(x_img)
        x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
        inputs = image_input

    output = layers.Dense(num_classes, activation="softmax", name="classification_output")(x)
    return models.Model(inputs=inputs, outputs=output, name="resnet50")

def build_densenet121(
    input_image_shape=(224, 224, 3), 
    num_metadata_features=0, 
    num_classes=3,
    dropout_rate=0.4,
    use_metadata=False,
):
    image_input = layers.Input(shape=input_image_shape, name="image_input")
    base_model = DenseNet121(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
        name="densenet121_backbone",
    )
    base_model.trainable = False

    x_img = _build_image_branch(base_model, dropout_rate)

    if use_metadata:
        meta_input = layers.Input(shape=(num_metadata_features,), name="meta_input")
        x_meta = layers.Dense(64, activation="relu", name="metadata_dense")(meta_input)
        x_meta = layers.Dropout(0.3, name="metadata_dropout")(x_meta)
        merged = layers.Concatenate(name="feature_fusion")([x_img, x_meta])
        x = layers.Dense(128, activation="relu", name="fusion_dense")(merged)
        x = layers.Dropout(dropout_rate, name="fusion_dropout")(x)
        inputs = [image_input, meta_input]
    else:
        x = layers.Dense(128, activation="relu", name="classifier_dense")(x_img)
        x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
        inputs = image_input

    output = layers.Dense(num_classes, activation="softmax", name="classification_output")(x)
    return models.Model(inputs=inputs, outputs=output, name="densenet121")


def build_model(
    model_name,
    input_image_shape, 
    num_metadata_features=0, 
    num_classes=3,
    dropout_rate=0.4,
    use_metadata=False,
):
    """Build one of the supported backbones under the same controlled setup."""
    model_name = model_name.lower()

    if model_name == "efficientnet":
        return build_efficientnet(input_image_shape, num_metadata_features, num_classes, dropout_rate, use_metadata)
    elif model_name == "resnet50":
        return build_resnet50(input_image_shape, num_metadata_features, num_classes, dropout_rate, use_metadata)
    elif model_name == "densenet121":
        return build_densenet121(input_image_shape, num_metadata_features, num_classes, dropout_rate, use_metadata)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Choose from 'efficientnet', 'resnet50', or 'densenet121'.")


def set_fine_tuning(model, model_name, trainable_layers=30):
    """
    Unfreeze only the upper part of the selected ImageNet backbone.

    Batch-normalization layers remain frozen because changing their running
    statistics with a small medical dataset can destabilize fine-tuning.
    """
    backbone = model.get_layer(f"{model_name}_backbone")
    backbone.trainable = True

    for layer in backbone.layers:
        layer.trainable = False

    # Unfreeze the last N non-BatchNorm layers.
    candidate_layers = [
        layer for layer in backbone.layers
        if not isinstance(layer, tf.keras.layers.BatchNormalization)
    ]
    for layer in candidate_layers[-trainable_layers:]:
        layer.trainable = True

    return model