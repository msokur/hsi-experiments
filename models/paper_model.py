import abc

import tensorflow as tf
import tensorflow.keras as keras

from models.model_randomness import get_dropout, get_initializers, set_tf_seed


class PaperModelBase:
    def __init__(self):
        self.conf = None
        self.num_of_labels = None
        self.kernel_initializer, self.bias_initializer = get_initializers()
        set_tf_seed()

    @abc.abstractmethod
    def get_model(self, shape: tuple, conf: dict, num_of_labels: int):
        pass

    def paper_model_base(self, input_, net, conv_round):
        net = self.get_1D_block(net=net, conv_round=conv_round)

        net = tf.keras.layers.Flatten()(net)

        net = get_dropout(net=net, dropout_value=self.conf["DROPOUT"])

        activation = 'sigmoid'
        number = 1
        if self.num_of_labels > 2:
            activation = None
            number = self.num_of_labels

        result = tf.keras.layers.Dense(number, activation=activation, kernel_initializer=self.kernel_initializer,
                                       bias_initializer=self.bias_initializer)(net)

        model = tf.keras.Model(
            inputs=[input_],
            outputs=[result]
        )

        return model

    def get_1D_block(self, net, conv_round):
        for r in conv_round:
            if net.shape[-2] == 1:
                break
            if net.shape[-2] == 2:
                net = self.get_conv1d(net=net, name=f"1D_{r}", kernel_size=2, strides=1)
                break
            else:
                net = self.get_conv1d(net=net, name=f"1D_{r}")

        return net

    def get_conv1d(self, net, name, kernel_size=3, strides=2):
        net = keras.layers.Conv1D(filters=35, kernel_size=kernel_size, padding='valid', activation='relu',
                                  kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                  name=f"{name}_1")(net)
        net = keras.layers.Conv1D(filters=35, kernel_size=kernel_size if net.shape[-2] > 1 else 1, strides=strides,
                                  padding='valid', activation='relu',
                                  kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                  name=f"{name}_2")(net)

        return net


class PaperModel1D(PaperModelBase):
    def get_model(self, shape: tuple, conf: dict, num_of_labels: int):
        self.conf = conf
        self.num_of_labels = num_of_labels
        input_ = tf.keras.layers.Input(shape=shape, name="title")

        net = tf.expand_dims(input_, axis=-1)
        conv_round = range(4)

        return self.paper_model_base(input_=input_, net=net, conv_round=conv_round)


class PaperModel3D(PaperModelBase):
    def get_model(self, shape: tuple, conf: dict, num_of_labels: int):
        self.conf = conf
        self.num_of_labels = num_of_labels
        input_ = tf.keras.layers.Input(shape=shape, name="title")

        conv_round = range(int(shape[0] / 2))

        net = tf.expand_dims(input_, axis=-1)

        for r in conv_round:
            if net.shape[-3] == 2:
                net = self.get_conv3d(net=net, name=f"3D_{r}", kernel_size=2, strides=1)
                break
            else:
                net = self.get_conv3d(net=net, name=f"3D_{r}")

        net = tf.keras.layers.Reshape((net.shape[-2], net.shape[-1]))(net)

        return self.paper_model_base(input_=input_, net=net, conv_round=conv_round)

    def get_conv3d(self, net, name, kernel_size=3, strides=2):
        net = keras.layers.Conv3D(filters=20, kernel_size=kernel_size, padding='valid', activation='relu',
                                  kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                  name=f"{name}_1")(net)
        net = keras.layers.Conv3D(filters=20, kernel_size=(1, 1, kernel_size), strides=(1, 1, strides), padding='valid',
                                  activation='relu', kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  name=f"{name}_2")(net)

        return net


if __name__ == "__main__":
    import os

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    shape_ = (92,)
    conf_ = {
        "DROPOUT": 0.1
    }
    labels = 3
    model1 = PaperModel1D()
    model_ = model1.get_model(shape=shape_, conf=conf_, num_of_labels=labels)
    model_.summary()
