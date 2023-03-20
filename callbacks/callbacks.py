import tensorflow as tf
import os


def callback_template_1(
        save_dir,
        checkpoint=True,
        reduce_lr_on_plateau=False,
        csv_logger=True,
        early_stopping=True,
        early_stopping_monitor="val_acc",
        early_stopping_patience=10,
        early_stopping_mode="auto",
):
    callbacks = []

    if checkpoint:
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(save_dir, "model.h5"),
                monitor='val_acc',
                mode='max',
                save_best_only=True,
                verbose=1
            )
        )

    if reduce_lr_on_plateau:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_acc",
                mode="max",
                patience=5,
                verbose=1,
                factor=0.5,
            )
        )

    if csv_logger:
        callbacks.append(
            tf.keras.callbacks.CSVLogger(os.path.join(save_dir, "log.csv"))
        )

    if early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor=early_stopping_monitor,
                patience=early_stopping_patience,
                verbose=1,
                mode=early_stopping_mode,
                restore_best_weights=True,
            )
        )

    return callbacks
