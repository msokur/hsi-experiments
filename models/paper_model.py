import abc

import tensorflow as tf
import tensorflow.keras as keras

from models.model_base import ModelBase
from models.model_randomness import get_dropout

from configuration.keys import ModelKeys as MK


class PaperModelBase(ModelBase):
    def get_model(self) -> keras.Model:
        if self.name is None:
            self.name = MK.PAPER_MODEL_NAME

        input_ = keras.layers.Input(shape=self.input_shape, name="input")

        return self._create_model(input_layer=input_)

    @abc.abstractmethod
    def _create_model(self, input_layer: keras.layers.Input) -> keras.Model:
        pass

    def _paper_model_base(self, input_, net, conv_round) -> keras.Model:
        net = self.__get_1D_block(net=net, conv_round=conv_round)

        net = keras.layers.Flatten(name="flatten_layer")(net)

        net = get_dropout(net=net, dropout_value=self.model_config[MK.DROPOUT], name="last_dropout_layer")

        activation = 'sigmoid'
        number = 1
        if self.num_of_output > 2:
            activation = None
            number = self.num_of_output

        result = keras.layers.Dense(number, activation=activation, kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer, name="predictions")(net)

        model = keras.Model(
            inputs=[input_],
            outputs=[result],
            name=self.name
        )

        return model

    def __get_1D_block(self, net, conv_round):
        for r in conv_round:
            if net.shape[-2] < 6:
                break
            else:
                net = self.__get_conv1d(net=net, name=f"1D_{r + 1}")

        return net

    def __get_conv1d(self, net, name, kernel_size=3, strides=2):
        net = keras.layers.Conv1D(filters=35, kernel_size=kernel_size, padding='valid', activation='relu',
                                  kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                  name=f"{name}.1_conv")(net)
        net = keras.layers.Conv1D(filters=35, kernel_size=kernel_size, strides=strides,
                                  padding='valid', activation='relu',
                                  kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                  name=f"{name}.2_conv")(net)

        return net


class PaperModel1D(PaperModelBase):
    def _create_model(self, input_layer: keras.layers.Input) -> keras.Model:
        net = tf.expand_dims(input_layer, axis=-1)
        conv_round = range(4)

        return self._paper_model_base(input_=input_layer, net=net, conv_round=conv_round)


class PaperModel3D(PaperModelBase):
    def _create_model(self, input_layer: keras.layers.Input) -> keras.Model:
        conv_round = range(int(self.input_shape[0] / 2))

        net = tf.expand_dims(input_layer, axis=-1)

        for r in conv_round:
            if net.shape[-2] < 6:
                break
            else:
                net = self.__get_conv3d(net=net, name=f"3D_{r + 1}")

        net = keras.layers.Reshape((net.shape[-4] * net.shape[-3] * net.shape[-2], net.shape[-1]))(net)

        return self._paper_model_base(input_=input_layer, net=net, conv_round=conv_round)

    def __get_conv3d(self, net, name, kernel_size=3, stride=2):
        net = keras.layers.Conv3D(filters=20, kernel_size=kernel_size, padding='valid', activation='relu',
                                  kernel_initializer=self.kernel_initializer, bias_initializer=self.bias_initializer,
                                  name=f"{name}.1_conv")(net)
        net = keras.layers.Conv3D(filters=20, kernel_size=(1, 1, kernel_size), strides=(1, 1, stride), padding='valid',
                                  activation='relu', kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  name=f"{name}.2_conv")(net)

        return net


if __name__ == "__main__":
    import os

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    shape_ = (3, 3, 92,)
    conf_ = {
        "DROPOUT": 0.1
    }
    labels = 3
    model1 = PaperModel3D(input_shape=shape_, model_config=conf_, num_of_output=labels)
    model_ = model1.get_model()
    model_.summary()
