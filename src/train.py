import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau  # update: needed for adaptive LR during fine-tuning
from tensorflow.keras.layers import BatchNormalization  # update: needed to identify and keep BatchNorm layers frozen

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


# update: NEW FUNCTION -- finds the backbone layers without needing models.py to
# hand us a base_model reference. It works by checking which layers are still
# marked trainable=False (only the backbone has this set, from models.py),
# then unfreezes the top % of them, keeping BatchNorm layers frozen throughout.
def _unfreeze_top_backbone_layers(model, fine_tune_at_percent=0.2):
    """
    Identifies backbone layers by checking which ones are still trainable=False
    (only the backbone has this set explicitly in models.py). Unfreezes the top
    `fine_tune_at_percent` fraction of those layers (the ones closest to the
    output), while keeping BatchNormalization layers frozen even within that
    unfrozen region, since their running stats were learned on ImageNet.

    Using a PERCENTAGE (not a fixed layer count) keeps this identical and fair
    across EfficientNetB3, ResNet50, and DenseNet121, since they have very
    different depths.
    """
    backbone_layers = [l for l in model.layers if not l.trainable]
    total = len(backbone_layers)

    if total == 0:
        print("[fine-tune] No frozen backbone layers found -- skipping unfreeze.")
        return

    fine_tune_at = int(total * (1 - fine_tune_at_percent))
    for layer in backbone_layers[:fine_tune_at]:
        layer.trainable = False
    for layer in backbone_layers[fine_tune_at:]:
        layer.trainable = not isinstance(layer, BatchNormalization)

    unfrozen = sum(l.trainable for l in backbone_layers)
    print(f"[fine-tune] Unfroze {unfrozen}/{total} backbone layers "
          f"(top {fine_tune_at_percent*100:.0f}%, BatchNorm layers kept frozen)")


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
    fine_tune=True,               # update: turns the new fine-tuning phase on/off
    fine_tune_epochs=20,          # update: how many epochs Phase 2 (fine-tune) runs
    fine_tune_lr=0.00001,         # update: much lower LR for Phase 2, so pretrained weights aren't wrecked
    fine_tune_at_percent=0.2,     # update: fraction of backbone unfrozen in Phase 2, same for every model
):
    """
    Compiles the model, configures early stopping and checkpoints, 
    and executes training on the provided datasets/generators.

    # update: Now runs in TWO phases when fine_tune=True (default):
    #   Phase 1 (warmup):   backbone stays frozen exactly as models.py built it,
    #                        trains for `epochs` at `learning_rate`. This is the
    #                        same behavior as the original single-phase code.
    #   Phase 2 (fine-tune): top `fine_tune_at_percent` of the backbone is unfrozen
    #                        and trained for `fine_tune_epochs` at `fine_tune_lr`.
    # All new arguments have defaults, so the existing call in cross_validation.py
    # does not need to change, and every architecture gets identical treatment.
    """
    
    # Default save path to model name if not provided
    if save_path is None:
        save_path = f"{model.name}.h5"
    
    # Ensure saving directory exists
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # ---------- Phase 1: Warmup (backbone frozen, same as original behavior) ----------
    print(f"\n[Phase 1] Warmup -- head only, backbone frozen ({epochs} epochs max)")  # update

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
    warmup_callbacks = [  # update: renamed from `callbacks` to `warmup_callbacks` to distinguish from Phase 2
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

    # 4. Fit model & Save best weights (handled via ModelCheckpoint)
    history_warmup = model.fit(  # update: renamed from `history` to `history_warmup`
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=warmup_callbacks,
    )

    # update: if fine-tuning is turned off, behave exactly like the original function
    if not fine_tune:
        return history_warmup

    # update: ---------- START OF NEW PHASE 2 BLOCK (fine-tuning) ----------
    print(f"\n[Phase 2] Fine-tuning -- top {fine_tune_at_percent*100:.0f}% of backbone "
          f"unfrozen ({fine_tune_epochs} epochs max)")

    # update: reload best Phase 1 weights before unfreezing, so Phase 2 starts
    # from the best-known state rather than wherever early stopping left off
    model.load_weights(save_path)

    # update: unfreeze the top % of backbone layers (BatchNorm stays frozen)
    _unfreeze_top_backbone_layers(model, fine_tune_at_percent=fine_tune_at_percent)

    # update: must recompile after changing any layer's trainable flag
    loss_fn_ft = SparseCategoricalFocalLoss(gamma=2.0, class_weight=class_weight)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss=loss_fn_ft,
        metrics=['accuracy']
    )

    fine_tune_callbacks = [  # update: Phase 2 gets its own callback list, including ReduceLROnPlateau
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    history_finetune = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=fine_tune_epochs,
        callbacks=fine_tune_callbacks,
    )
    # update: ---------- END OF NEW PHASE 2 BLOCK ----------

    return {"warmup": history_warmup, "fine_tune": history_finetune}  # update: now returns both phases' histories