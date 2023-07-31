import tensorflow.keras as keras
import trainers.trainer_easy as trainer_easy
from configuration.keys import TrainerKeys as TK, DataLoaderKeys as DLK


class TrainerEasySeveralOutputs(trainer_easy.TrainerEasy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compile_model(self, model: keras.Model) -> keras.Model:
        metric_dict = self.CONFIG_TRAINER[TK.CUSTOM_OBJECTS]
        METRICS = [
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ]
        for key in metric_dict.keys():
            METRICS.append(metric_dict[key]["metric"](num_classes=len(self.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]),
                                                      **metric_dict[key]["args"]))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.CONFIG_TRAINER[TK.LEARNING_RATE]),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=METRICS
        )

        return model
