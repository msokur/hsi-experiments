import tensorflow.keras as keras
import numpy as np
import os
from shutil import rmtree

from trainers.trainer_base import Trainer
from configuration.keys import TrainerKeys as TK
from configuration.parameter import (
    MODEL_BATCH_SIZE,
)


class TrainerEasy(Trainer):
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
            for_tuning=self.CONFIG_TRAINER[TK.SMALLER_DATASET])

        '''-------TRAINING---------'''

        history = model.fit(
            # x=train_generator,
            # validation_data=valid_generator,
            x=train_dataset,
            validation_data=valid_dataset,
            epochs=self.CONFIG_TRAINER[TK.EPOCHS],
            verbose=2,
            initial_epoch=initial_epoch,
            batch_size=MODEL_BATCH_SIZE,
            callbacks=self.get_callbacks(),
            use_multiprocessing=True,
            class_weight=class_weights,
            workers=int(os.cpu_count()))

        self.save_history(history)

        rmtree(self.batch_path)

        return model, history

    def compile_model(self, model: keras.Model) -> keras.Model:
        metric_dict = self.CONFIG_TRAINER[TK.CUSTOM_OBJECTS]
        METRICS = [
            keras.metrics.BinaryAccuracy(name="accuracy"),
        ]
        for key in metric_dict.keys():
            METRICS.append(metric_dict[key]["metric"](**metric_dict[key]["args"]))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.CONFIG_TRAINER[TK.LEARNING_RATE]),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=METRICS,
        )

        return model

    def get_restored_model(self) -> tuple[keras.Model, int]:
        print('!!!!!!!!!!!!We restore model!!!!!!!!!!!!')
        search_path = os.path.join(self.log_dir, 'checkpoints')
        all_checkpoints = [os.path.join(search_path, d) for d in os.listdir(search_path) if
                           os.path.isdir(os.path.join(search_path, d))]
        sorted(all_checkpoints)
        all_checkpoints = np.array(all_checkpoints)

        initial_epoch = int(all_checkpoints[-1].split('-')[-1])

        model = keras.models.load_model(all_checkpoints[-1],
                                        custom_objects=self.CONFIG_TRAINER[TK.CUSTOM_OBJECTS_LOAD],
                                        compile=True)

        return model, initial_epoch

    def get_new_model(self) -> keras.Model:
        model = self.CONFIG_TRAINER[TK.MODEL](shape=self.get_output_shape(), conf=self.CONFIG_TRAINER[TK.MODEL_CONFIG],
                                              num_of_labels=len(self.labels_to_train))
        model = self.compile_model(model)
        return model

    def get_model(self) -> tuple[keras.Model, int]:
        initial_epoch = 0
        if self.CONFIG_TRAINER[TK.RESTORE]:
            model, initial_epoch = self.get_restored_model()
        else:
            model = self.get_new_model()

        return model, initial_epoch


if __name__ == '__main__':
    pass
