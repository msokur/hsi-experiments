from tensorflow import keras
import trainers.trainer_easy as trainer_easy


class TrainerEasySeveralOutputs(trainer_easy.TrainerEasy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compile_model(self, model):
        metric_dict = self.trainer["CUSTOM_OBJECTS"]
        METRICS = [
            keras.metrics.SparseCategoricalAccuracy(),
        ]
        for key in metric_dict.keys():
            METRICS.append(metric_dict[key]["metric"](num_classes=len(self.loader["LABELS_TO_TRAIN"]),
                                                      **metric_dict[key]["args"]))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.trainer["LEARNING_RATE"]),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=METRICS
        )

        return model
