from tensorflow import keras

import config
import trainer_easy
import models.model_3d as model_3d
from util.tf_metric_multiclass import F1_score


class TrainerEasySeveralOutputs(trainer_easy.TrainerEasy):
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
            keras.metrics.BinaryAccuracy(name='accuracy'),
            # keras.metrics.AUC(name='auc'),
            # tf_metrics.f1_m
            # specificity_m
        ]
        # model = inception_model()

        model.compile(
            # optimizer=keras.optimizers.Adam(lr=config.LEARNING_RATE, clipnorm=1.),
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            # optimizer=keras.optimizers.RMSprop(lr=config.LEARNING_RATE),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[keras.metrics.SparseCategoricalAccuracy(),
                     F1_score(num_classes=config.NUMBER_OF_CLASSES_TO_TRAIN, average='weighted')]
            # metrics=METRICS
            # weighted_metrics=WEIGHTED_METRICS
        )

        return model
    
    def get_easy_model(self):
        model = model_3d.paper_model()
        model = self.compile_model(model)
        return model
    