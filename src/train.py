import os
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)


class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """
    Custom Focal Loss that accepts sparse integer targets (0, 1, 2).
    """

    def __init__(
        self,
        gamma=2.0,
        class_weight=None,
        name="sparse_categorical_focal_loss"
    ):
        super().__init__(name=name)

        self.gamma = gamma

        if class_weight is not None:
            # Convert:
            # {0: w0, 1: w1, 2: w2}
            # into:
            # [w0, w1, w2]

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

        # Convert labels to integer
        y_true = tf.cast(
            y_true,
            tf.int32
        )

        y_true = tf.reshape(
            y_true,
            [-1]
        )

        # Prevent numerical instability
        y_pred = tf.clip_by_value(
            y_pred,
            1e-7,
            1.0 - 1e-7
        )

        # Convert sparse labels to one-hot
        num_classes = tf.shape(
            y_pred
        )[-1]

        y_true_one_hot = tf.one_hot(
            y_true,
            depth=num_classes
        )

        # Probability of the correct class
        p_t = tf.reduce_sum(
            y_true_one_hot * y_pred,
            axis=-1
        )

        # Focal Loss
        focal_loss = (
            -tf.pow(
                1.0 - p_t,
                self.gamma
            )
            * tf.math.log(p_t)
        )

        # Apply class weights
        if self.class_weight is not None:

            alpha_t = tf.gather(
                self.class_weight,
                y_true
            )

            focal_loss = (
                focal_loss * alpha_t
            )

        return tf.reduce_mean(
            focal_loss
        )


def train_model(
    model,
    train_ds,
    val_ds,
    epochs=30,
    learning_rate=1e-4,
    steps_per_epoch=None,
    validation_steps=None,
    save_path=None,
    class_weight=None,
    fine_tune=True,
    fine_tune_epochs=10,
    fine_tune_lr=1e-5,
):
    """
    Two-stage transfer learning.

    Stage 1:
        Pretrained backbone remains frozen.

    Stage 2:
        Last 40 backbone layers are unfrozen
        for fine-tuning.

    BatchNormalization layers remain frozen
    during fine-tuning.
    """

    # =========================================================
    # SAVE PATH
    # =========================================================

    if save_path is None:
        save_path = f"{model.name}.weights.h5"

    save_dir = os.path.dirname(
        save_path
    )

    if save_dir:
        os.makedirs(
            save_dir,
            exist_ok=True
        )

    # =========================================================
    # LOSS
    # =========================================================

    loss_fn = SparseCategoricalFocalLoss(
        gamma=2.0,
        class_weight=class_weight
    )

    # =========================================================
    # STAGE 1
    # =========================================================

    print("\n" + "=" * 65)
    print("STAGE 1: TRANSFER LEARNING")
    print("=" * 65)

    print(
        f"Learning rate: {learning_rate}"
    )

    print(
        f"Maximum epochs: {epochs}"
    )

    print(
        "Pretrained backbone: FROZEN"
    )

    # ---------------------------------------------------------
    # Compile Stage 1
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
    # Stage 1 checkpoint
    # ---------------------------------------------------------

    stage1_path = save_path

    stage1_callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),

        ModelCheckpoint(
            filepath=stage1_path,
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
    # Train Stage 1
    # ---------------------------------------------------------

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=stage1_callbacks
    )

    # =========================================================
    # LOAD BEST STAGE 1 MODEL
    # =========================================================

    if os.path.exists(stage1_path):

        print(
            "\nLoading best Stage 1 weights..."
        )

        model.load_weights(
            stage1_path
        )

    # =========================================================
    # STAGE 2: FINE-TUNING
    # =========================================================

    if fine_tune:

        print("\n" + "=" * 65)
        print("STAGE 2: DENSENET121 FINE-TUNING")
        print("=" * 65)

        # -----------------------------------------------------
        # Get backbone
        # -----------------------------------------------------

        if not hasattr(
            model,
            "base_model"
        ):

            raise AttributeError(
                "The model does not contain "
                "'base_model'. "
                "Your models.py must expose "
                "the pretrained backbone."
            )

        base_model = model.base_model

        # -----------------------------------------------------
        # Freeze everything first
        # -----------------------------------------------------

        for layer in base_model.layers:
            layer.trainable = False

        # -----------------------------------------------------
        # Unfreeze last 40 layers
        # -----------------------------------------------------

        for layer in base_model.layers[-40:]:
            layer.trainable = True

        # -----------------------------------------------------
        # Keep BatchNormalization frozen
        # -----------------------------------------------------

        for layer in base_model.layers:

            if isinstance(
                layer,
                tf.keras.layers.BatchNormalization
            ):
                layer.trainable = False

        # -----------------------------------------------------
        # Display trainable layers
        # -----------------------------------------------------

        trainable_layers = sum(
            layer.trainable
            for layer in base_model.layers
        )

        total_layers = len(
            base_model.layers
        )

        print(
            f"DenseNet121 backbone: "
            f"{trainable_layers}/{total_layers} "
            f"layers trainable"
        )

        print(
            f"Fine-tuning learning rate: "
            f"{fine_tune_lr}"
        )

        print(
            f"Fine-tuning epochs: "
            f"{fine_tune_epochs}"
        )

        # -----------------------------------------------------
        # Recompile with LOWER learning rate
        # -----------------------------------------------------

        fine_optimizer = tf.keras.optimizers.Adam(
            learning_rate=fine_tune_lr
        )

        model.compile(
            optimizer=fine_optimizer,
            loss=loss_fn,
            metrics=["accuracy"]
        )

        # =====================================================
        # FINE-TUNING CHECKPOINT
        # =====================================================

        fine_tune_path = save_path.replace(
            ".weights.h5",
            ".finetuned.weights.h5"
        )

        stage2_callbacks = [

            EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),

            ModelCheckpoint(
                filepath=fine_tune_path,
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

        # =====================================================
        # FINE-TUNE
        # =====================================================

        history_fine = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            epochs=fine_tune_epochs,
            callbacks=stage2_callbacks
        )

        # =====================================================
        # LOAD BEST FINE-TUNED MODEL
        # =====================================================

        if os.path.exists(
            fine_tune_path
        ):

            print(
                "\nLoading best fine-tuned weights..."
            )

            model.load_weights(
                fine_tune_path
            )

        print("\nFine-tuning completed.")

        return history_fine

    # =========================================================
    # NO FINE-TUNING
    # =========================================================

    print(
        "\nFine-tuning disabled."
    )

    print(
        "Training completed."
    )

    return history