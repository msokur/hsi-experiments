import tensorflow.keras as keras
import trainers.trainer_easy as trainer_easy


class TrainerEasySeveralOutputs(trainer_easy.TrainerEasy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_parameters_for_compile(self):
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # TODO, check if from logits?
        metric_dict = self.CONFIG_TRAINER["CUSTOM_OBJECTS"]
        raw_metrics = []
        
        for key in metric_dict.keys():
            raw_metrics.append(metric_dict[key]["metric"](num_classes=len(self.CONFIG_DATALOADER["LABELS_TO_TRAIN"]), **metric_dict[key]["args"]))
                
        return loss, raw_metrics
