import tensorflow.keras as keras
import numpy as np
import os

from trainers.trainer_base import Trainer
from configuration.keys import TrainerKeys as TK, DataLoaderKeys as DLK, CrossValidationKeys as CVK
from configuration.parameter import (
    MODEL_BATCH_SIZE,
)


class TrainerBinary(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_process(self):
        self.logging_and_copying()

        '''-------MODEL---------'''
        if self.mirrored_strategy is not None:
            try:
                with self.mirrored_strategy.scope():
                    model, initial_epoch = self.get_model()
            except Exception as e:
                raise e
        else:
            model, initial_epoch = self.get_model()
        model.summary()

        '''-------DATASET---------'''

        train_dataset, valid_dataset, class_weights = self.get_datasets(
            for_tuning=self.config.CONFIG_TRAINER[TK.USE_SMALLER_DATASET])

        '''-------TRAINING---------'''

        history = model.fit(
            x=train_dataset,
            validation_data=valid_dataset,
            epochs=self.config.CONFIG_TRAINER[TK.EPOCHS],
            verbose=2,
            initial_epoch=initial_epoch,
            batch_size=MODEL_BATCH_SIZE,
            callbacks=self.get_callbacks(),
            use_multiprocessing=True,
            class_weight=class_weights,
            workers=int(os.cpu_count()))

        self.save_history(history)

        if self.config.CONFIG_CV[CVK.MODE] == "RUN":
            self.dataset.delete_batches(batch_path=self.batch_path)

        return model, history

    def get_loss_and_metrics(self):
        loss = keras.losses.BinaryCrossentropy()
        metric_dict = self.config.CONFIG_TRAINER[TK.CUSTOM_OBJECTS]
        raw_metrics = [
            keras.metrics.BinaryAccuracy(name="accuracy")
        ]

        non_weightable_metrics = [
            # add in non_weightable_metrics metrics that should not be affected by class_weights or sample_weights
            # For example absolut values, like tp, tn, fp, fn (True Positives, ....)
        ]

        for key in metric_dict.keys():
            raw_metrics.append(metric_dict[key]["metric"](**metric_dict[key]["args"]))

        return loss, raw_metrics, non_weightable_metrics

    def get_restored_model(self) -> tuple[keras.Model, int]:
        print('!!!!!!!!!!!!We restore model!!!!!!!!!!!!')
        search_path = os.path.join(self.log_dir, 'checkpoints')
        all_checkpoints = [os.path.join(search_path, d) for d in os.listdir(search_path) if
                           os.path.isdir(os.path.join(search_path, d))]
        sorted(all_checkpoints)
        all_checkpoints = np.array(all_checkpoints)

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

    def get_model(self) -> tuple[keras.Model, int]:
        initial_epoch = 0
        if self.config.CONFIG_TRAINER[TK.RESTORE]:
            model, initial_epoch = self.get_restored_model()
        else:
            model = self.get_new_model()

        return model, initial_epoch


if __name__ == '__main__':
    pass
