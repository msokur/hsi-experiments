from tensorflow import keras
import trainers.trainer_easy as trainer_easy


class TrainerEasySeveralOutputs(trainer_easy.TrainerEasy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compile_model(self, model):
        METRICS = [
            keras.metrics.SparseCategoricalAccuracy(),
        ]
        for key in self.trainer["CUSTOM_OBJECTS"].keys():
            METRICS.append(self.trainer["CUSTOM_OBJECTS"][key](num_classes=len(self.loader["LABELS_TO_TRAIN"]),
                                                               average='weighted'))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.trainer["LEARNING_RATE"]),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=METRICS
        )

        return model
