from tensorflow import keras
import numpy as np
import os

import trainers.trainer_base as trainer_base


class TrainerEasy(trainer_base.Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compile_model(self, model):
        METRICS = [
            keras.metrics.BinaryAccuracy(name='accuracy'),
        ]
        METRICS += self.trainer["CUSTOM_OBJECTS"]

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.trainer["LEARNING_RATE"]),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=METRICS,
        )

        return model

    def get_restored_model(self):
        print('!!!!!!!!!!!!We restore model!!!!!!!!!!!!')
        search_path = os.path.join(self.log_dir, 'checkpoints')
        all_checkpoints = [os.path.join(search_path, d) for d in os.listdir(search_path) if
                           os.path.isdir(os.path.join(search_path, d))]
        sorted(all_checkpoints)
        all_checkpoints = np.array(all_checkpoints)

        initial_epoch = int(all_checkpoints[-1].split('-')[-1])

        if self.mirrored_strategy is not None:
            with self.mirrored_strategy.scope():
                model = keras.models.load_model(all_checkpoints[-1],
                                                custom_objects=self.trainer["CUSTOM_OBJECTS"],
                                                compile=True)
        else:
            model = keras.models.load_model(all_checkpoints[-1], self.trainer["CUSTOM_OBJECTS"])

        model = self.compile_model(model)

        return model, initial_epoch

    def get_easy_model(self):
        model = self.trainer["MODEL"](shape=self.__get_output_shape(), conf=self.trainer["MODEL_CONFIG"],
                                      num_of_labels=len(self.loader["LABELS_TO_TRAIN"]))
        model = self.compile_model(model)
        return model

    def get_model(self):
        initial_epoch = 0
        if self.trainer["RESTORE"]:
            model, initial_epoch = self.get_restored_model()
        else:
            model = self.get_easy_model()

        return model, initial_epoch


if __name__ == '__main__':
    pass
