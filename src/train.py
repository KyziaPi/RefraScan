import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(
    model,
    train_ds,
    val_ds,
    epochs=50,
    learning_rate=0.0001,
    steps_per_epoch=None,
    validation_steps=None,
    save_path="models/best_model.h5"
):
    """
    Compiles the model, configures early stopping and checkpoints, 
    and executes training on the provided datasets/generators.
    """
    # Ensure saving directory exists
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # 1. Compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
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
        callbacks=callbacks
    )

    return history