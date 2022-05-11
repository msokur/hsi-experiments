from tensorflow import keras
import numpy as np
import os

import model_3d
import config
import trainer_base
import tf_metrics

class TrainerEasy(trainer_base.Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compile_model(self, model):
        METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.Recall(name='sensitivity')
        ]
        WEIGHTED_METRICS = [
            #keras.metrics.BinaryAccuracy(name='accuracy'),
            # keras.metrics.AUC(name='auc'),
            tf_metrics.f1_m
            # specificity_m
        ]
        # model = inception_model()

        model.compile(
            # optimizer=keras.optimizers.Adam(lr=config.LEARNING_RATE, clipnorm=1.),
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            # optimizer=keras.optimizers.RMSprop(lr=config.LEARNING_RATE),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=keras.metrics.BinaryAccuracy(name='accuracy'),
            # metrics=METRICS,
            weighted_metrics=WEIGHTED_METRICS
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
                                                custom_objects=config.CUSTOM_OBJECTS,
                                                compile=True)
        else:
            model = keras.models.load_model(all_checkpoints[-1], custom_objects=config.CUSTOM_OBJECTS)

        model = self.compile_model(model)

        return model, initial_epoch

    def get_easy_model(self):
        model = model_3d.inception3d_model()
        model = self.compile_model(model)
        return model

    def get_model(self):
        initial_epoch = 0
        if config.RESTORE_MODEL:
            model, initial_epoch = self.get_restored_model()
        else:
            # model = model_3d.paper_model()
            model = self.get_easy_model()

        return model, initial_epoch


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    #train(except_indexes=['2019_09_04_12_43_40_', '2020_05_28_15_20_27_', '2019_07_12_11_15_49_', '2020_05_15_12_43_58_'])
