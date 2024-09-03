import os

import keras
import psutil

from .custom_callbacks.tensorboard_callback import TensorboardCallback

from configuration.keys import TrainerKeys as TK


def get_callbacks(callback_configs: dict, checkpoint_dir: str, debug: bool):
    callbacks = [_get_model_checkpoint(checkpoint_config=callback_configs[TK.MODEL_CHECKPOINT],
                                       save_path=checkpoint_dir)]

    if callback_configs[TK.EARLY_STOPPING]["enable"]:
        callbacks.append(get_early_stopping(early_stopping_config=callback_configs[TK.EARLY_STOPPING]))

    if debug:
        callbacks.append(_get_tensorboard())

    return callbacks


def _get_model_checkpoint(checkpoint_config: dict, save_path: str):
    try:
        if checkpoint_config["save_best_only"]:
            name = "cp_best"
        else:
            name = "cp-{epoch:04d}"
        checkpoint_path = os.path.join(save_path, name)

        checkpoints_callback = keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=checkpoint_config["monitor"],
            verbose=1,
            save_best_only=checkpoint_config["save_best_only"],
            mode=checkpoint_config["mode"])

        return checkpoints_callback
    except TypeError as e:
        print("Check your configurations for the model checkpoint callback!")
        raise TypeError(e)


def get_early_stopping(early_stopping_config: dict):
    try:
        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor=early_stopping_config["monitor"],
            mode=early_stopping_config["mode"],
            min_delta=early_stopping_config["min_delta"],
            patience=early_stopping_config["patience"],
            verbose=1,
            restore_best_weights=early_stopping_config["restore_best_weights"])

        return early_stopping_callback
    except TypeError as e:
        print("Check your configurations for the early stopping callback!")
        raise TypeError(e)


def _get_tensorboard():
    return TensorboardCallback(process=psutil.Process(os.getpid()))
