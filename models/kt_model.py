import abc

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.model_randomness import get_dropout, set_tf_seed


class InceptionTunerModelBase(kt.HyperModel):
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
            metrics = [keras.metrics.BinaryAccuracy(name='accuracy')]
            if self.custom_metrics is not None:
                for key in self.custom_metrics.keys():
                    metrics.append(self.custom_metrics[key]["metric"](**self.custom_metrics[key]["args"]))
        else:
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = [keras.metrics.SparseCategoricalAccuracy()]
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

        class_weights = None
        if hp.Boolean("class_weights"):
            class_weights = class_weight

        return model.fit(
            clss_weight=class_weights,
            **kwargs
        )

    def model(self):
        input_ = layers.Input(shape=self.shape, name="title")

        net = tf.expand_dims(input_, axis=-1)
        for bl in range(self.hp.Int("num_blocks", **self.config["NUM_BLOCK"])):
            net = self.model_block(input_=net, name=f"bl{bl}")

        net = layers.Flatten()(net)

        for fc in range(self.hp.Int("num_fc", **self.config["NUM_DENSE"])):
            net = layers.Dense(self.hp.Int(f"fc_{fc}", **self.config["DENSE_UNITS"]),
                               activation=self.get_activations(name=f"fc_{fc}"))(net)

            if self.hp.Boolean(f"fc_{fc}_dropout"):
                net = get_dropout(net=net,
                                  dropout_value=self.hp.Float(f"fc_{fc}_dropout_value", **self.config["DROPOUT"]))

        activation = "sigmoid"
        number = 1
        if self.num_of_classes > 2:
            activation = None
            number = self.num_of_classes

        result = layers.Dense(number, activation=activation)

        model = keras.Model(
            inputs=[input_],
            outputs=[result]
        )

        return model

    def model_block(self, input_, name):
        branches = []

        b1_name = self.wrap_name(name, "b1")
        b1_f_name = self.wrap_name(b1_name, "f")
        branch1 = self.get_conv_layer(input_=input_, name=b1_name, f_name=b1_f_name, k_name="", kernel_size=1)
        branch1 = self.wrap_layer(branch1, b1_name)
        branches.append(branch1)

        for i in range(self.hp(self.wrap_name(name, "num_branches"), **self.config["NUM_BRANCHES"])):
            branch = self.model_branch(input_, name, i)
            branches.append(branch)

        b_last_name = self.wrap_name(name, "b_last")
        b_last_f_name = self.wrap_name(b_last_name, "f")
        b_last_k_name = self.wrap_name(b_last_name, "k")
        branch_last = self.get_max_pool(input_=input_, name=b_last_name)
        branch_last = self.get_conv_layer(input_=branch_last, name=b_last_name, f_name=b_last_f_name,
                                          k_name=b_last_k_name, kernel_size=1 if self.config["WITH_ONES"] else None)
        branch_last = self.wrap_layer(branch_last, b_last_name)
        branches.append(branch_last)

        net = layers.concatenate(branches)

        return net

    def model_branch(self, input_, name, idx):
        branch = input_
        for idx_sub in range(1, 3):
            b_name = self.wrap_name(name, f"b{idx + 1}_{idx_sub}")
            f_name = self.wrap_name(b_name, "f")
            k_name = self.wrap_name(b_name, "k")
            branch = self.get_conv_layer(input_=branch, name=b_name, f_name=f_name, k_name=k_name,
                                         kernel_size=1 if self.config["WITH_ONES"] & idx_sub == 1 else None)
            branch = self.wrap_layer(branch, b_name)

        return branch

    @abc.abstractmethod
    def get_conv_layer(self, input_, name, f_name, k_name, kernel_size=None):
        pass

    @abc.abstractmethod
    def get_max_pool(self, input_, name):
        pass

    @staticmethod
    def wrap_name(layer_name, name):
        return layer_name + "." + name

    def get_activations(self, name):
        return self.hp.Choice(self.wrap_name(name, "activation"), self.config["ACTIVATION"])

    def get_optimizer(self):
        optimizer = self.hp.Choice("optimizer", [key for key in self.config["OPTIMIZER"].keys()])
        lr = self.hp.Float("lr", **self.config["LEARNING_RATE"])
        return self.config[optimizer](learning_rate=lr)

    def wrap_layer(self, layer, name):
        order_activation_batch_norm = self.hp.Choice(self.wrap_name(name, "mode"), ['without_batch_norm',
                                                                                    'batch_norm_activation',
                                                                                    'activation_batch_norm'])

        batch_name = self.wrap_name(name, 'batch_norm')
        activation_name = self.wrap_name(name, 'activation')
        dropout_name = self.wrap_name(name, 'dropout')

        if order_activation_batch_norm == "batch_norm_activation":
            layer = layers.BatchNormalization(name=batch_name)(layer)  # no batch_normalization

        layer = layers.Activation(self.get_activations(name), name=activation_name)(layer)

        if order_activation_batch_norm == "activation_batch_norm":
            layer = layers.BatchNormalization(name=batch_name)(layer)  # batch_normalization than activation

        if self.hp.Boolean(self.wrap_name(name, "dr")):
            layer = layers.Dropout(self.hp.Float(self.wrap_name(name, "dr_val"), **self.config["DROPOUT"]),
                                   name=dropout_name)(layer)

        return layer

    def declare_hyperparameters(self, hp):
        pass


class InceptionTunerModel3D(InceptionTunerModelBase):
    def get_conv_layer(self, input_, name, f_name, k_name, kernel_size=None):
        return layers.Conv3D(
            filters=self.hp.Int(f_name, **self.config["BLOCK_FILTERS"]),
            kernel_size=self.hp.Int(k_name, **self.config["BLOCK_KERNEL_SIZE"]) if kernel_size is None
            else kernel_size,
            padding="same",
            name=name)(input_)

    def get_max_pool(self, input_, name):
        return layers.MaxPooling3D(pool_size=3, strides=1, padding="same",
                                   name=self.wrap_name(name, 'max_pool'))(input_)


class InceptionTunerModel1D(InceptionTunerModelBase):
    def get_conv_layer(self, input_, name, f_name, k_name, kernel_size=None):
        return layers.Conv1D(
            filters=self.hp.Int(f_name, **self.config["BLOCK_FILTERS"]),
            kernel_size=self.hp.Int(k_name, **self.config["BLOCK_KERNEL_SIZE"]) if kernel_size is None
            else kernel_size,
            padding="same",
            name=name)(input_)

    def get_max_pool(self, input_, name):
        return layers.MaxPooling1D(pool_size=3, strides=1, padding="same",
                                   name=self.wrap_name(name, 'max_pool'))(input_)
