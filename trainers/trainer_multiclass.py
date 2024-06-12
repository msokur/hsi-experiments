import tensorflow.keras as keras
from trainers.trainer_binary import TrainerBinary
from configuration.keys import TrainerKeys as TK, DataLoaderKeys as DLK


class TrainerMulticlass(TrainerBinary):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_loss_and_metrics(self):
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # TODO, check if from logits?
        metric_dict = self.config.CONFIG_TRAINER[TK.CUSTOM_OBJECTS]
        raw_metrics = [
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.F1Score(**self.config.CONFIG_TRAINER[TK.F1_SCORE])
        ]

        non_weightable_metrics = [
            # add in non_weightable_metrics metrics that should not be affected by class_weights or sample_weights
            # For example absolut values, like tp, tn, fp, fn (True Positives, ....)
        ]

        for key in metric_dict.keys():
            raw_metrics.append(metric_dict[key]["metric"](
                num_classes=len(self.config.CONFIG_DATALOADER[DLK.LABELS_TO_TRAIN]), **metric_dict[key]["args"]))

        return loss, raw_metrics, non_weightable_metrics
