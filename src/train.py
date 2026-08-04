import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class SparseCategoricalFocalLoss(tf.keras.losses.Loss):
  """Custom Focal Loss that accepts sparse integer targets (0, 1, 2)."""

  def __init__(self, gamma=2.0, class_weight=None, name="sparse_categorical_focal_loss"):
    super().__init__(name=name)
    self.gamma = gamma
    
    if class_weight is not None:
      # Convert class_weight dict {0: w0, 1: w1, 2: w2} to Tensor [w0, w1, w2]
      weights = [class_weight[i] for i in sorted(class_weight.keys())]
      self.class_weight = tf.constant(weights, dtype=tf.float32)
    else:
      self.class_weight = None

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

    # Apply class weights (alpha_t) directly to the loss tensor
    if self.class_weight is not None:
      alpha_t = tf.gather(self.class_weight, y_true)
      focal_loss = focal_loss * alpha_t

    return tf.reduce_mean(focal_loss)

 #updated: added unfreeze_resnet_stage function to selectively unfreeze ResNet50 layers while keeping BatchNorm frozen 
def unfreeze_resnet_stage(model, stage_prefixes=("conv5_block3",)):
    """Unfreezes ResNet50 layers matching the given stage prefix, keeping BatchNorm frozen."""
    unfrozen = []
    for layer in model.layers:
        if layer.name.startswith(stage_prefixes):
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
                unfrozen.append(layer.name)
    print(f"Unfroze {len(unfrozen)} layers: {unfrozen}")
    return model

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
    patience=7 #updated: added patience parameter for early stopping
):
    """
    Compiles the model, configures early stopping and checkpoints, 
    and executes training on the provided datasets/generators.
    """
    
    # Default save path to model name if not provided
    if save_path is None:
        save_path = f"{model.name}.keras"
    
    # Ensure saving directory exists
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # 1. Instantiate Focal Loss with class_weight injected directly
    loss_fn = SparseCategoricalFocalLoss(gamma=2.0, class_weight=class_weight)

    # 2. Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']
    )

    # 3. Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience, #updated: added patience parameter for early stopping
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]

    # 4. Fit model & Save best weights (handled via ModelCheckpoint)
    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
    )

    return history