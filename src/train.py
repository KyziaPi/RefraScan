import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
  """Custom Focal Loss that accepts sparse integer targets (0, 1, 2)."""

  def __init__(self, gamma=2.0, name="sparse_categorical_focal_loss"):
    super().__init__(name=name)
    self.gamma = gamma

  def call(self, y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_true = tf.reshape(y_true, [-1])

    # Clip predictions to prevent numerical instability log(0)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

    # Convert integer labels to one-hot vectors
    num_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(y_true, depth=num_classes)

    # Extract prediction probability corresponding to the true class
    p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)

    # Calculate Focal Loss: - (1 - p_t)^gamma * log(p_t)
    focal_loss = -tf.pow(1.0 - p_t, self.gamma) * tf.math.log(p_t)

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
    class_weights=None,
):
    """
    Compiles the model, configures early stopping and checkpoints, 
    and executes training on the provided datasets/generators.
    """
    
    # Default save path to model name if not provided
    if save_path is None:
        save_path = f"{model.name}.h5"
    
    # Ensure saving directory exists
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    loss_fn = SparseCategoricalFocalLoss(gamma=2)

    # 1. Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    # 2. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # 3. Fit model & Save best weights (handled via ModelCheckpoint)
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights
    )

    return history