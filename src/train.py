import os
from xml.parsers.expat import model
import tensorflow as tf

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, class_weight=None,
                 name="sparse_categorical_focal_loss"):
        super().__init__(name=name)
        self.gamma = gamma

        if class_weight is not None:
            weights = [class_weight[i] for i in sorted(class_weight.keys())]
            self.class_weight = tf.constant(weights, dtype=tf.float32)
        else:
            self.class_weight = None

    def call(self, y_true, y_pred):

        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.reshape(y_true, [-1])

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        num_classes = tf.shape(y_pred)[-1]
        y_true_onehot = tf.one_hot(y_true, depth=num_classes)

        pt = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)

        loss = -tf.pow(1.0 - pt, self.gamma) * tf.math.log(pt)

        if self.class_weight is not None:
            alpha = tf.gather(self.class_weight, y_true)
            loss *= alpha

        return tf.reduce_mean(loss)


###########################################################
# Fine-tuning function
###########################################################

def unfreeze_model(model, model_name):
    """
    Selectively unfreezes the last block(s) of the backbone for fine-tuning.

    Note: each build_* function constructs its backbone with
    `input_tensor=image_input`, so the backbone's internal layers (e.g.
    DenseNet121's conv/bn/relu layers) are flattened directly into the outer
    functional model's `model.layers` list rather than appearing as a single
    nested tf.keras.Model instance. So we operate on `model.layers` directly
    instead of searching for a nested backbone sub-model. The backbone layers
    are already frozen from Stage 1 (build_model set base_model.trainable =
    False), and the custom head layers are already trainable — we only need
    to selectively re-enable training on the last backbone block.
    """

    print(f"\nFine-tuning {model_name}")

    if model_name.lower() == "densenet121":

        for layer in model.layers:

            # keep BatchNorm layers frozen even inside the unfrozen block
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                continue

            # unfreeze the entire last Dense Block
            if (
                layer.name.startswith("conv5")
                or layer.name.startswith("pool5")
                or layer.name.startswith("relu")
            ):
                layer.trainable = True

    elif model_name.lower() == "efficientnet":

        # unfreeze approximately the last 30% of layers for fine-tuning
        total_layers = len(model.layers)

        for layer in model.layers[int(total_layers * 0.7):]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    elif model_name.lower() == "resnet50":

        for layer in model.layers:

            # keep BatchNorm layers frozen even inside the unfrozen block
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                continue

            # unfreeze the entire last Conv Block
            if layer.name.startswith("conv5"):
                layer.trainable = True

    trainable_count = sum(layer.trainable for layer in model.layers)

    print(f"Trainable layers: {trainable_count}")

    print("\n========== TRAINABLE LAYERS ==========")

    for layer in model.layers:
        if layer.trainable:
            print(layer.name)

    print("======================================\n")

    return model


###########################################################
# Train
###########################################################

def train_model(
    model,
    train_ds,
    val_ds,
    epochs=40,
    learning_rate=1e-4,
    steps_per_epoch=None,
    validation_steps=None,
    save_path=None,
    class_weight=None,
    patience=10,
):

    if save_path is None:
        save_path = model.name + ".h5"

    dirname = os.path.dirname(save_path)

    if dirname:
        os.makedirs(dirname, exist_ok=True)

    loss_fn = SparseCategoricalFocalLoss(
        gamma=2.0,
        class_weight=class_weight
    )

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=1e-5,
        ),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),

        ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )

    ]

    history = model.fit(

        train_ds,

        validation_data=val_ds,

        epochs=epochs,

        steps_per_epoch=steps_per_epoch,

        validation_steps=validation_steps,

        callbacks=callbacks

    )

    return history