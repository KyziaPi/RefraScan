import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Custom Focal Loss that accepts sparse integer targets (0, 1, 2)."""

    def __init__(
        self,
        gamma=2.0,
        class_weight=None,
        name="sparse_categorical_focal_loss"
    ):
        super().__init__(name=name)

        self.gamma = gamma

        if class_weight is not None:
            # Convert class_weight dict {0: w0, 1: w1, 2: w2}
            # to Tensor [w0, w1, w2]
            weights = [
                class_weight[i]
                for i in sorted(class_weight.keys())
            ]

            self.class_weight = tf.constant(
                weights,
                dtype=tf.float32
            )
        else:
            self.class_weight = None

    def call(self, y_true, y_pred):

        y_true = tf.cast(y_true, tf.int32)
        y_true = tf.reshape(y_true, [-1])

        # Clip predictions to prevent numerical instability
        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        # Convert integer labels to one-hot vectors
        num_classes = tf.shape(y_pred)[-1]

        y_true_one_hot = tf.one_hot(
            y_true,
            depth=num_classes
        )

        # Extract probability corresponding to true class
        p_t = tf.reduce_sum(
            y_true_one_hot * y_pred,
            axis=-1
        )

        # Calculate Focal Loss
        focal_loss = (
            -tf.pow(1.0 - p_t, self.gamma)
            * tf.math.log(p_t)
        )

        # Apply class weights
        if self.class_weight is not None:

            alpha_t = tf.gather(
                self.class_weight,
                y_true
            )

            focal_loss = focal_loss * alpha_t

        return tf.reduce_mean(focal_loss)


def train_model(
    model,
    train_ds,
    val_ds,
    epochs=50,
    learning_rate=0.0001,
    steps_per_epoch=None,
    validation_steps=None,
    save_path=None,
    class_weight=None,
    fine_tune=True,
    fine_tune_epochs=10,
    fine_tune_lr=1e-5
):
    """
    Compiles and trains the model.

    Stage 1:
        Train the classification head with the pretrained
        backbone frozen.

    Stage 2:
        Fine-tune the last 40 layers of the pretrained backbone.

    Model weights are saved using .weights.h5.
    """

    # ---------------------------------------------------------
    # Default save path
    # ---------------------------------------------------------

    if save_path is None:
        save_path = f"{model.name}.weights.h5"

    # Ensure saving directory exists
    dir_name = os.path.dirname(save_path)

    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Create Focal Loss
    # ---------------------------------------------------------

    loss_fn = SparseCategoricalFocalLoss(
        gamma=2.0,
        class_weight=class_weight
    )

    # ---------------------------------------------------------
    # 2. Compile model
    # ---------------------------------------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"]
    )

    # ---------------------------------------------------------
    # 3. Callbacks for Stage 1
    # ---------------------------------------------------------

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            filepath=save_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # ---------------------------------------------------------
    # 4. Stage 1 Training
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("STAGE 1: TRAINING CLASSIFICATION HEAD")
    print("=" * 60)

    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print("DenseNet121 backbone: FROZEN")

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks
    )

    # ---------------------------------------------------------
    # 5. Fine-Tuning Stage
    # ---------------------------------------------------------

    if fine_tune:

        print("\n" + "=" * 60)
        print("STAGE 2: DENSENET121 FINE-TUNING")
        print("=" * 60)

        # Load the best Stage 1 weights
        if os.path.exists(save_path):
            print(f"Loading best Stage 1 weights: {save_path}")
            model.load_weights(save_path)

        # -----------------------------------------------------
        # Get DenseNet121 backbone
        # -----------------------------------------------------

        base_model = getattr(model, "base_model", None)

        if base_model is None:
            raise ValueError(
                "DenseNet121 backbone not found. "
                "Make sure models.py contains: "
                "model.base_model = base_model"
            )

        print(f"Backbone found: {base_model.name}")

        # -----------------------------------------------------
        # Freeze ALL backbone layers first
        # -----------------------------------------------------

        for layer in base_model.layers:
            layer.trainable = False

        # -----------------------------------------------------
        # Unfreeze last 40 layers
        # -----------------------------------------------------

        for layer in base_model.layers[-40:]:
            layer.trainable = True

        # -----------------------------------------------------
        # Keep BatchNormalization layers frozen
        # -----------------------------------------------------

        for layer in base_model.layers:

            if isinstance(
                layer,
                tf.keras.layers.BatchNormalization
            ):
                layer.trainable = False

        # -----------------------------------------------------
        # Display fine-tuning information
        # -----------------------------------------------------

        trainable_layers = sum(
            layer.trainable
            for layer in base_model.layers
        )

        total_layers = len(base_model.layers)

        print(
            f"DenseNet121 Fine-Tuning: "
            f"{trainable_layers}/{total_layers} "
            f"layers are trainable."
        )

        # -----------------------------------------------------
        # Recompile with lower learning rate
        # -----------------------------------------------------

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=fine_tune_lr
        )

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=["accuracy"]
        )

        print(f"Fine-tuning learning rate: {fine_tune_lr}")
        print(f"Fine-tuning epochs: {fine_tune_epochs}")

        # -----------------------------------------------------
        # Fine-tuning callbacks
        # -----------------------------------------------------

        fine_tune_callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),

            ModelCheckpoint(
                filepath=save_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            ),

            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]

        # -----------------------------------------------------
        # Fine-tune model
        # -----------------------------------------------------

        history_fine = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            epochs=fine_tune_epochs,
            callbacks=fine_tune_callbacks
        )

        print("\nFine-tuning completed.")

        # Load the BEST fine-tuned weights
        if os.path.exists(save_path):
            model.load_weights(save_path)
            print(f"Best weights loaded from: {save_path}")

        return history_fine

    # ---------------------------------------------------------
    # Return Stage 1 history if fine-tuning disabled
    # ---------------------------------------------------------

    return history