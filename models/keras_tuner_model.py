import keras_tuner
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models.keras_tuner_model import *
import config


class KerasTunerModel(keras_tuner.HyperModel):

    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)
        self.hp = None

    def wrap_name(self, layer_name, name):
        return layer_name + "_" + name

    def get_hp_activations(self, name):
        return self.hp.Choice(self.wrap_name(name, "activation"),
                              ["relu", "tanh", "selu", "exponential", "elu"])

    def wrap_layer(self, layer, name):
        order_activation_batch_norm = self.hp.Int(self.wrap_name(name, "mode"), 0, 2)

        if order_activation_batch_norm == 0:  # no batch_normalization
            layer = layers.Activation(self.get_hp_activations(name))(layer)
        if order_activation_batch_norm == 1:  # activation than batch_normalization
            layer = layers.Activation(self.get_hp_activations(name))(layer)
            layer = layers.BatchNormalization()(layer)
        if order_activation_batch_norm == 2:  # batch_normalization than activation
            layer = layers.BatchNormalization()(layer)
            layer = layers.Activation(self.get_hp_activations(name))(layer)

        if self.hp.Boolean(self.wrap_name(name, "dropout_if")):
            layer = layers.Dropout(self.hp.Float(self.wrap_name(name, "dropout_value"),
                                                 min_value=0.1,
                                                 max_value=0.9,
                                                 step=0.1))(layer)

        return layer

    # inception_branch(self, input_, name):


    def inception3d_block(self, input_, name):
        input_ = tf.expand_dims(input_, axis=-1)

        branch1 = layers.Conv3D(filters=self.hp.Int(self.wrap_name(name,"b1_filters"), min_value=16, max_value=128, step=16),
                                # kernel_size=self.hp.Int("kernel_b1", min_value=1, max_value=7, step=2),
                                kernel_size=1,
                                padding="same")(input_)
        branch1 = self.wrap_layer(branch1, self.wrap_name(name, "b1"))

        branch2 = layers.Conv3D(filters=self.hp.Int(self.wrap_name(name,"b2_1_filters"), min_value=16, max_value=128, step=16),
                                kernel_size=self.hp.Int(self.wrap_name(name,"b2_1_kernel"), min_value=1, max_value=7, step=2),
                                padding="same")(input_)
        branch2 = self.wrap_layer(branch2, self.wrap_name(name, "b2_1"))
        branch2 = layers.Conv3D(filters=self.hp.Int(self.wrap_name(name,"b2_2_filters"), min_value=16, max_value=128, step=16),
                                kernel_size=self.hp.Int(self.wrap_name(name,"b2_2_kernel"), min_value=1, max_value=7, step=2),
                                padding="same")(branch2)
        branch2 = self.wrap_layer(branch2, self.wrap_name(name, "b2_2"))

        branch3 = layers.Conv3D(filters=self.hp.Int(self.wrap_name(name, "b3_1_filters"), min_value=16, max_value=128, step=16),
                                kernel_size=self.hp.Int(self.wrap_name(name,"b3_1_kernel"), min_value=1, max_value=7, step=2),
                                padding="same")(input_)
        branch3 = self.wrap_layer(branch3, self.wrap_name(name, "b3_1"))
        branch3 = layers.Conv3D(filters=self.hp.Int(self.wrap_name(name,"b3_2_filters"), min_value=16, max_value=128, step=16),
                                kernel_size=self.hp.Int(self.wrap_name(name,"b3_2_kernel"), min_value=1, max_value=7, step=2),
                                padding="same")(branch3)
        branch3 = self.wrap_layer(branch3, self.wrap_name(name, "b3_2"))


        branch_last = layers.MaxPooling3D(pool_size=3, strides=1, padding="same")(input_)
        branch_last = layers.Conv3D(filters=self.hp.Int(self.wrap_name(name,"b_last_filters"), min_value=16, max_value=128, step=16),
                                kernel_size=self.hp.Int(self.wrap_name(name,"b_last_kernel"), min_value=1, max_value=7, step=2),
                                padding='same')(branch_last)
        branch_last = self.wrap_layer(branch_last, self.wrap_name(name, "b_last"))

        net = layers.concatenate([branch1, branch2, branch3, branch_last])

        return net

    def inception3d_model(self):
        input_ = layers.Input(
            shape=(config._3D_SIZE[0],
                   config._3D_SIZE[1],
                   config.OUTPUT_SIGNATURE_X_FEATURES),
            name="title"
        )

        net = self.inception3d_block(input_, "inc_block_0")

        net = layers.Flatten()(net)

        if self.hp.Boolean("dropout_last"):
            net = layers.Dropout(self.hp.Float("dropout_value", min_value=0.1, max_value=0.9, step=0.1))(net)
        result = layers.Dense(1, activation='sigmoid')(net)

        model = tf.keras.Model(
            inputs=[input_],
            outputs=[result]
        )

        return model

    def build(self, hp):
        self.hp = hp
        model = self.inception3d_model()

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            # optimizer=keras.optimizers.RMSprop(lr=config.LEARNING_RATE),
            loss=keras.losses.BinaryCrossentropy(),
            metrics=keras.metrics.BinaryAccuracy(name='accuracy'),
            # Objective is one of the metrics.
            # metrics=[keras.metrics.MeanAbsoluteError()],
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        model.summary()

        return model.fit(
            **kwargs
        )
