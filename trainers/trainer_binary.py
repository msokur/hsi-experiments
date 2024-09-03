from typing import Dict

import tensorflow.keras as keras
import os

from trainers.trainer_base import Trainer
from .utils import get_loss_and_metrics
from configuration.keys import (
    TrainerKeys as TK,
    DataLoaderKeys as DLK,
    PathKeys as PK
)
from configuration.parameter import (
    MODEL_BATCH_SIZE,
)


class TrainerBinary(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_process(self, train_log_dir: str, datasets: tuple, class_weights: Dict[int, float], batch_path: str):
        # -------MODEL---------
        if self.mirrored_strategy is not None:
            try:
                with self.mirrored_strategy.scope():
                    model, initial_epoch = self.get_model(train_log_dir=train_log_dir)
            except Exception as e:
                raise e
        else:
            model, initial_epoch = self.get_model(train_log_dir=train_log_dir)
        model.summary()

        # -------TRAINING---------
        history = model.fit(
            x=datasets[0],
            validation_data=datasets[1],
            epochs=self.config.CONFIG_TRAINER[TK.EPOCHS],
            verbose=2,
            initial_epoch=initial_epoch,
            batch_size=MODEL_BATCH_SIZE,
            callbacks=self.get_callbacks(train_log_dir=train_log_dir),
            use_multiprocessing=True,
            class_weight=class_weights,
            workers=int(os.cpu_count()))

        return model, history

    def get_model(self, train_log_dir: str) -> tuple[keras.Model, int]:
        initial_epoch = 0
        if self.config.CONFIG_TRAINER[TK.RESTORE]:
            model, initial_epoch = self.get_restored_model(train_log_dir=train_log_dir)
        else:
            model = self.get_new_model()

        return model, initial_epoch

    def get_restored_model(self, train_log_dir: str) -> tuple[keras.Model, int]:
        print('!!!!!!!!!!!!We restore model!!!!!!!!!!!!')
        search_path = os.path.join(train_log_dir, self.config.CONFIG_CV_PATH[PK.CHECKPOINT_FOLDER])
        all_checkpoints = [os.path.join(search_path, d) for d in os.listdir(search_path) if
                           os.path.isdir(os.path.join(search_path, d))]
        sorted(all_checkpoints)

        initial_epoch = int(all_checkpoints[-1].split('-')[-1])

        model = keras.models.load_model(all_checkpoints[-1],
                                        custom_objects=self.config.CONFIG_TRAINER[TK.CUSTOM_OBJECTS_LOAD],
                                        compile=True)

        return model, initial_epoch

    def get_new_model(self) -> keras.Model:
        model = self.config.CONFIG_TRAINER[TK.MODEL](input_shape=self.get_output_shape(),
                                                     model_config=self.config.CONFIG_TRAINER[TK.MODEL_CONFIG],
                                                     num_of_output=len(self.config.CONFIG_DATALOADER[
                                                                           DLK.LABELS_TO_TRAIN])).get_model()
        model = self.compile_model(model)
        return model

    def compile_model(self, model):
        loss, metrics, weighted_metrics = (
            get_loss_and_metrics(label_count=len(self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]),
                                 custom_metrics=self.config.CONFIG_TRAINER[TK.CUSTOM_OBJECTS],
                                 with_sample_weights=self.config.CONFIG_TRAINER[TK.WITH_SAMPLE_WEIGHTS]))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.CONFIG_TRAINER["LEARNING_RATE"]),
            loss=loss,
            metrics=metrics,
            weighted_metrics=weighted_metrics
        )

        return model


if __name__ == '__main__':
    pass
