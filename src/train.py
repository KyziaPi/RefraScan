import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

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
    fine_tune_lr=1e-5,
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
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=save_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
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

    # Fine-tuning Stage
    if fine_tune:
       print("\nStarting fine-tuning...\n")

       # Load the best weights from Stage 1
       model.load_weights(save_path)

       # Find the pretrained backbone
       base_model = None

       for layer in model.layers:
          if isinstance(layer, tf.keras.Model):
             base_model = layer
             break

       if base_model is None:
           raise ValueError("Could not locate DenseNet121 backbone.")

       # Freeze all backbone layers
       for layer in base_model.layers:
           layer.trainable = False

       # Unfreeze only the last 40 layers
       for layer in base_model.layers[-40:]:
           layer.trainable = True

       # Keep BatchNormalization layers frozen
       for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

       trainable = sum(layer.trainable for layer in base_model.layers)
       total = len(base_model.layers)
       
       print(f"DenseNet121 Fine-Tuning: {trainable}/{total} layers are trainable.")

       # Recompile with a lower learning rate
       optimizer = tf.keras.optimizers.Adam(
           learning_rate=fine_tune_lr
       )

       model.compile(
           optimizer=optimizer,
           loss=loss_fn,
           metrics=["accuracy"]
        )

       history_fine = model.fit(
           train_ds,
           steps_per_epoch=steps_per_epoch,
           validation_data=val_ds,
           validation_steps=validation_steps,
           epochs=fine_tune_epochs,
           callbacks=callbacks,
        )

       # Save the final fine-tuned model
       model.save_weights(save_path)

       
       return history_fine

    return history