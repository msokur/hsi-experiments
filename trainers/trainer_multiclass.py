import tensorflow.keras as keras
from trainers.trainer_binary import TrainerEasy
from configuration.keys import TrainerKeys as TK, DataLoaderKeys as DLK


class TrainerEasySeveralOutputs(TrainerEasy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compile_model(self, model: keras.Model) -> keras.Model:
        metric_dict = self.config.CONFIG_TRAINER[TK.CUSTOM_OBJECTS]
        METRICS = [
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ]
        for key in metric_dict.keys():
            METRICS.append(metric_dict[key]["metric"](num_classes=len(self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]),
                                                      **metric_dict[key]["args"]))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.CONFIG_TRAINER[TK.LEARNING_RATE]),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=METRICS
        )

        return model

    def get_loss_and_metrics(self):
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # TODO, check if from logits?
        metric_dict = self.config.CONFIG_TRAINER["CUSTOM_OBJECTS"]
        raw_metrics = []

        for key in metric_dict.keys():
            raw_metrics.append(metric_dict[key]["metric"](
                num_classes=len(self.config.CONFIG_DATALOADER["LABELS_TO_TRAIN"]), **metric_dict[key]["args"]))

        return loss, raw_metrics
