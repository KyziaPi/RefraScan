# src/explainability.py
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


CLASS_NAMES = ["Emmetropia", "Myopia", "Hyperopia"]


def make_gradcam_heatmap(model, image, metadata=None, class_index=None, layer_name=None):
    """
    Generate a Grad-CAM heatmap for an image-only or image+metadata model.

    `image` should already be preprocessed and have shape (1, H, W, 3).
    For the thesis, the target class should normally be the model's predicted
    class unless a specific true-class explanation is desired.
    """
    if layer_name is None:
        # Support both nested-backbone models and flat Keras applications
        # models (e.g., the current ResNet50 final model).
        backbone_candidates = [
            layer for layer in model.layers
            if "backbone" in layer.name
        ]

        if backbone_candidates:
            backbone = backbone_candidates[0]
            conv_layers = [
                layer for layer in backbone.layers
                if isinstance(layer, tf.keras.layers.Conv2D)
                and getattr(layer.output, "shape", None) is not None
                and len(layer.output.shape) == 4
            ]
            if not conv_layers:
                raise ValueError("No convolutional layer found inside the CNN backbone.")
            layer_name = conv_layers[-1].name

        else:
            # The current final model exposes ResNet50 layers directly.
            conv_layers = [
                layer for layer in model.layers
                if isinstance(layer, tf.keras.layers.Conv2D)
                and getattr(layer.output, "shape", None) is not None
                and len(layer.output.shape) == 4
            ]
            if not conv_layers:
                raise ValueError("No convolutional layer found for Grad-CAM.")
            layer_name = conv_layers[-1].name

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )

    # Support both image-only and image+age models. For image+age, the caller
    # should provide the actual scaled age value for the selected record.
    if isinstance(model.input, list):
        if metadata is None:
            raise ValueError("metadata is required for Grad-CAM on an image+metadata model.")
        model_inputs = [image, tf.convert_to_tensor(metadata, dtype=tf.float32)]
    else:
        model_inputs = image

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(model_inputs)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
    conv_outputs = conv_outputs[0]
    pooled_grads = pooled_grads[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(class_index)


def overlay_gradcam(original_rgb, heatmap, alpha=0.4):
    """Resize the heatmap and overlay it on the original fundus image."""
    heatmap = cv2.resize(heatmap, (original_rgb.shape[1], original_rgb.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_rgb.astype(np.uint8), 1 - alpha, heatmap_color, alpha, 0)
    return overlay


def show_gradcam(original_rgb, heatmap, true_class=None, predicted_class=None):
    """Display the original image, Grad-CAM heatmap, and overlay."""
    overlay = overlay_gradcam(original_rgb, heatmap)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original_rgb)
    plt.axis("off")
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.axis("off")
    plt.title("Grad-CAM")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.axis("off")
    title = "Grad-CAM Overlay"
    if true_class is not None and predicted_class is not None:
        title += f"\nTrue: {CLASS_NAMES[true_class]} | Pred: {CLASS_NAMES[predicted_class]}"
    plt.title(title)
    plt.tight_layout()
    plt.show()
