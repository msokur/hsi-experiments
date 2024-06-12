import abc

import keras_tuner as kt
from keras import layers, losses, metrics

from models.model_randomness import set_tf_seed, get_initializers

from configuration.keys import TunerModelKeys as TMK


class KtModelBase(kt.HyperModel):
    def __init__(self, input_shape: tuple, model_config: dict, num_of_labels: int, custom_metrics=None, name=None,
                 tunable=True):
        super().__init__(name, tunable)
        self.hp = None
        self.input_shape = input_shape
        self.model_config = model_config
        self.num_of_classes = num_of_labels
        self.custom_metrics = custom_metrics
        self.kernel_initializer, self.bias_initializer = get_initializers()
        set_tf_seed()

    def build(self, hp):
        self.hp = hp
        model = self._get_model()

        if self.num_of_classes == 2:
            loss = losses.BinaryCrossentropy(),
            metrics_ = [metrics.BinaryAccuracy(name="accuracy")]
            if self.custom_metrics is not None:
                for key in self.custom_metrics.keys():
                    metrics_.append(self.custom_metrics[key]["metric"](**self.custom_metrics[key]["args"]))
        else:
            loss = losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics_ = [metrics.SparseCategoricalAccuracy(name="accuracy")]
            if self.custom_metrics is not None:
                for key in self.custom_metrics.keys():
                    metrics_.append(self.custom_metrics[key]["metric"](num_classes=self.num_of_classes,
                                                                       **self.custom_metrics[key]["args"]))

        print("Keras tuner model metrics:", metrics_)

        model.compile(
            optimizer=self._get_optimizer(),
            loss=loss,
            metrics=metrics_
        )

        return model

    def fit(self, hp, model, class_weight=None, *args, **kwargs):
        model.summary()

        return model.fit(
            *args,
            class_weight=class_weight if hp.Boolean("class_weights") else None,
            **kwargs
        )

    def _get_model(self):
        input_ = layers.Input(shape=self.input_shape, name="input")
        input_expand = layers.Reshape(target_shape=input_.shape[1:] + (1,))(input_)

        return self._model(input_layer=input_, net=input_expand)

    @abc.abstractmethod
    def _model(self, input_layer: layers.Input, net: layers.Reshape):
        pass

    def _get_activations(self, name):
        return self.hp.Choice(self._wrap_name(name, "activation"),
                              [key for key in self.model_config[TMK.ACTIVATION].keys()])

    def _get_optimizer(self):
        optimizer = self.hp.Choice("optimizer", [key for key in self.model_config[TMK.OPTIMIZER].keys()])
        lr = self.hp.Float("lr", **self.model_config[TMK.LEARNING_RATE])
        return self.model_config[TMK.OPTIMIZER][optimizer](learning_rate=lr)

    def _wrap_layer(self, layer, name):
        order_activation_batch_norm = self.hp.Choice(self._wrap_name(name, "mode"), ["without_batch_norm",
                                                                                     "batch_norm_activation",
                                                                                     "activation_batch_norm"])

        batch_name = self._wrap_name(name, "batch_norm")
        activation_name = self._wrap_name(name, "activation")
        dropout_name = self._wrap_name(name, "dropout")

        if order_activation_batch_norm == "batch_norm_activation":
            layer = layers.BatchNormalization(name=batch_name)(layer)

        layer = layers.Activation(self._get_activations(name), name=activation_name)(layer)

        if order_activation_batch_norm == "activation_batch_norm":
            layer = layers.BatchNormalization(name=batch_name)(layer)

        if self.hp.Boolean(self._wrap_name(name, "dr")):
            layer = layers.Dropout(self.hp.Float(self._wrap_name(name, "dr_val"),
                                                 **self.model_config[TMK.DROPOUT]), name=dropout_name)(layer)

        return layer

    @staticmethod
    def _wrap_name(layer_name, name):
        return layer_name + "." + name

    def declare_hyperparameters(self, hp):
        pass
