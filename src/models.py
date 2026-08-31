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
):
    image_input = layers.Input(shape=input_image_shape, name="image_input")
    base_model = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
    )
    base_model._name = "efficientnetb3_backbone"
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
        x_meta = layers.Dense(16, activation="relu", name="metadata_dense")(meta_input)
        x_meta = layers.Dropout(0.2, name="metadata_dropout")(x_meta)
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
    Progressively unfreeze the upper part of the image backbone.

    The backbone may appear as a nested Keras model or may be flattened
    into the outer Functional model when constructed with input_tensor.
    BatchNormalization layers remain frozen during fine-tuning.
    """

    # ------------------------------------------------------------------
    # 1. Try to retrieve the backbone as a nested model.
    # ------------------------------------------------------------------
    backbone_name = f"{model_name}_backbone"

    try:
        backbone = model.get_layer(backbone_name)
        backbone_layers = backbone.layers
    except ValueError:
        # ------------------------------------------------------------------
        # 2. Keras may flatten the backbone into the outer Functional model.
        #    In that case, everything before global_average_pooling belongs
        #    to the image backbone.
        # ------------------------------------------------------------------
        backbone_layers = []

        for layer in model.layers:
            if layer.name == "global_average_pooling":
                break

            # Exclude the input layer.
            if not isinstance(layer, tf.keras.layers.InputLayer):
                backbone_layers.append(layer)

        if not backbone_layers:
            raise ValueError(
                f"Could not identify the {model_name} backbone layers."
            )

    # ------------------------------------------------------------------
    # 3. Freeze the entire backbone first.
    # ------------------------------------------------------------------
    for layer in backbone_layers:
        layer.trainable = False

    # ------------------------------------------------------------------
    # 4. Select the last N non-BatchNorm backbone layers.
    # ------------------------------------------------------------------
    candidate_layers = [
        layer
        for layer in backbone_layers
        if not isinstance(layer, tf.keras.layers.BatchNormalization)
    ]

    if trainable_layers <= 0:
        return model

    trainable_layers = min(trainable_layers, len(candidate_layers))

    # ------------------------------------------------------------------
    # 5. Unfreeze only the upper N non-BatchNorm layers.
    # ------------------------------------------------------------------
    for layer in candidate_layers[-trainable_layers:]:
        layer.trainable = True

    print(
        f"Fine-tuning {model_name}: "
        f"unfroze the upper {trainable_layers} non-BatchNorm backbone layers."
    )

    return model