import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
    """Sparse focal loss for the three-class refractive-error target."""

def __init__(self, gamma=2.0, class_weight=None, name="sparse_categorical_focal_loss"):
    super().__init__(name=name)
    self.gamma = gamma
    self.class_weight_dict = (
        {int(k): float(v) for k, v in class_weight.items()}
        if class_weight is not None
        else None
    )
    self.class_weight = None
    if self.class_weight_dict is not None:
        weights = [self.class_weight_dict[i] for i in sorted(self.class_weight_dict)]
        self.class_weight = tf.constant(weights, dtype=tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "class_weight": self.class_weight_dict,
        })
        return config

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        p_t = tf.reduce_sum(y_one_hot * y_pred, axis=-1)
        loss = -tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)

        if self.class_weight is not None:
            loss *= tf.gather(self.class_weight, y_true)

        return tf.reduce_mean(loss)


def train_model(
    model,
    train_ds,
    val_ds=None,
    epochs=30,
    learning_rate=1e-4,
    steps_per_epoch=None,
    validation_steps=None,
    save_path=None,
    class_weight=None,
):
    """Compile and train one fold using the fixed experimental strategy."""

    # Default save path to model name if not provided
    if save_path is None:
        save_path = f"{model.name}.keras"
    
    # Ensure saving directory exists
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    
    # 1. Instantiate Focal Loss with class_weight injected directly
    loss_fn = SparseCategoricalFocalLoss(gamma=2.0, class_weight=class_weight)

    # 2. Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    if val_ds is not None:
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=7,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=save_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
        ]

    # 3. Fit model & Save the complete model (handled via ModelCheckpoint)
        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_ds,
            validation_steps=validation_steps,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
        )
    else:
        # Final development-data training uses the already-selected epoch count
        # and learning rate. No holdout data is used for early stopping.
        checkpoint = ModelCheckpoint(
            filepath=save_path,
            save_best_only=False,
            save_weights_only=False,
            verbose=1,
        )
        history = model.fit(
            train_ds,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=[checkpoint],
            verbose=1,
        )
    return history