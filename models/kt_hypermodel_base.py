import abc

import keras_tuner as kt
import tensorflow.keras as keras
from keras import layers

from models.model_randomness import set_tf_seed


class KtModelBase(kt.HyperModel):
    def __init__(self, shape: tuple, conf: dict, num_of_labels: int, custom_metrics=None, name=None, tunable=True):
        super().__init__(name, tunable)
        self.hp = None
        self.shape = shape
        self.config = conf
        self.num_of_classes = num_of_labels
        self.custom_metrics = custom_metrics
        set_tf_seed()

    def build(self, hp):
        self.hp = hp
        model = self.model()

        if self.num_of_classes == 2:
            loss = keras.losses.BinaryCrossentropy(),
            metrics = [keras.metrics.BinaryAccuracy(name="accuracy")]
            if self.custom_metrics is not None:
                for key in self.custom_metrics.keys():
                    metrics.append(self.custom_metrics[key]["metric"](**self.custom_metrics[key]["args"]))
        else:
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = [keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
            if self.custom_metrics is not None:
                for key in self.custom_metrics.keys():
                    metrics.append(self.custom_metrics[key]["metric"](num_classes=self.num_of_classes,
                                                                      **self.custom_metrics[key]["args"]))

        print("Keras tuner model metrics:", metrics)

        model.compile(
            optimizer=self.get_optimizer(),
            loss=loss,
            metrics=metrics
        )

        return model

    def fit(self, hp, model, class_weight=None, *args, **kwargs):
        model.summary()

        return model.fit(
            *args,
            class_weight=class_weight if hp.Boolean("class_weights") else None,
            **kwargs
        )

    @abc.abstractmethod
    def model(self):
        pass

    def get_activations(self, name):
        return self.hp.Choice(self.wrap_name(name, "activation"), self.config["ACTIVATION"])

    def get_optimizer(self):
        optimizer = self.hp.Choice("optimizer", [key for key in self.config["OPTIMIZER"].keys()])
        lr = self.hp.Float("lr", **self.config["LEARNING_RATE"])
        return self.config["OPTIMIZER"][optimizer](learning_rate=lr)

    def wrap_layer(self, layer, name):
        order_activation_batch_norm = self.hp.Choice(self.wrap_name(name, "mode"), ['without_batch_norm',
                                                                                    'batch_norm_activation',
                                                                                    'activation_batch_norm'])

        batch_name = self.wrap_name(name, 'batch_norm')
        activation_name = self.wrap_name(name, 'activation')
        dropout_name = self.wrap_name(name, 'dropout')

        if order_activation_batch_norm == "batch_norm_activation":
            layer = layers.BatchNormalization(name=batch_name)(layer)

        layer = layers.Activation(self.get_activations(name), name=activation_name)(layer)

        if order_activation_batch_norm == "activation_batch_norm":
            layer = layers.BatchNormalization(name=batch_name)(layer)

        if self.hp.Boolean(self.wrap_name(name, "dr")):
            layer = layers.Dropout(self.hp.Float(self.wrap_name(name, "dr_val"), **self.config["DROPOUT"]),
                                   name=dropout_name)(layer)

        return layer

    @staticmethod
    def wrap_name(layer_name, name):
        return layer_name + "." + name

    def declare_hyperparameters(self, hp):
        pass
