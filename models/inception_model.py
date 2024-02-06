import abc

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import activations

from models.model_base import ModelBase
from models.model_randomness import get_dropout

from configuration.keys import ModelKeys as MK

FILTERS = [4, [6, 8], [1, 2]]
FILTERS_LAST = 2
KERNEL_SIZE = [1, [1, 3], [1, 5]]
KERNEL_SIZE_LAST = 1
POOL_SIZE = 3
POOL_STRIDES = 1


class InceptionModelBase(ModelBase):
    def get_model(self):
        if self.name is None:
            self.name = MK.INCEPTION_MODEL_NAME

        input_ = keras.layers.Input(
            shape=self.input_shape, name=self.name
        )

        net = self.__inception_block(input_=input_, factor=self.model_config[MK.INCEPTION_FACTOR],
                                     with_batch_norm=self.model_config[MK.WITH_BATCH_NORM])

        return self.__inception_base(input_=input_, net=net)

    def __inception_block(self, input_, factor=16, with_batch_norm=False):
        input_ = tf.expand_dims(input_, axis=-1)

        branches = []
        for idx, filter_, kernel_size in zip(range(len(FILTERS)), FILTERS, KERNEL_SIZE):
            if isinstance(filter_, int):
                name = f"{idx + 1}_conv"
                branch = self._get_conv_layer(filters=factor * filter_, kernel_size=kernel_size, net=input_,
                                              name=name)

                if with_batch_norm:
                    branch = self.__get_batch_norm(branch=branch, name=name)
            else:
                branch = input_
                for sub_idx, filter_sub, kernel_size_sub in zip(range(len(FILTERS)), filter_, kernel_size):
                    name = f"{idx + 1}.{sub_idx + 1}_conv"
                    branch = self._get_conv_layer(filters=factor * filter_sub, kernel_size=kernel_size_sub, net=branch,
                                                  name=name)

                    if with_batch_norm:
                        branch = self.__get_batch_norm(branch=branch, name=name)
            branches.append(branch)

        branch = self._get_max_pooling_layer(input_=input_)
        max_name = "conv_after_max_pooling"
        branch = self._get_conv_layer(filters=factor * FILTERS_LAST, kernel_size=KERNEL_SIZE_LAST, net=branch,
                                      name=max_name)

        if with_batch_norm:
            branch = self.__get_batch_norm(branch=branch, name=max_name)
        branches.append(branch)

        net = keras.layers.concatenate(branches)

        return net

    def __inception_base(self, input_, net):
        net = keras.layers.Flatten(name="flatten_layer")(net)
        net = get_dropout(net=net, dropout_value=self.model_config["DROPOUT"], name="last_dropout_layer")

        activation = 'sigmoid'
        number = 1
        if self.num_of_output > 2:
            activation = None
            number = self.num_of_output
        result = keras.layers.Dense(number, activation=activation, kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer, name="predictions")(net)

        model = keras.Model(
            inputs=[input_],
            outputs=[result]
        )

        return model

    @abc.abstractmethod
    def _get_conv_layer(self, filters, kernel_size, net, name):
        pass

    @abc.abstractmethod
    def _get_max_pooling_layer(self, input_):
        pass

    def __get_batch_norm(self, branch, name: str):
        if len(self.input_shape) > 1:
            name = "3D_" + name
        else:
            name = "1D_" + name
        branch = keras.layers.BatchNormalization(name=f"{name}_batch_normalization")(branch)
        branch = keras.layers.Activation(activation=activations.relu, name=f"{name}_activation")(branch)
        return branch


class InceptionModel3D(InceptionModelBase):
    def _get_conv_layer(self, filters, kernel_size, net, name):
        return keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, padding="same", activation='relu',
                                   kernel_initializer=self.kernel_initializer,
                                   bias_initializer=self.bias_initializer,
                                   name=f"3D_{name}")(net)

    def _get_max_pooling_layer(self, input_):
        return keras.layers.MaxPooling3D(pool_size=POOL_SIZE, strides=POOL_STRIDES, padding="same",
                                         name="3D_max_pooling_layer")(input_)


class InceptionModel1D(InceptionModelBase):
    def _get_conv_layer(self, filters, kernel_size, net, name):
        return keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same", activation='relu',
                                   kernel_initializer=self.kernel_initializer,
                                   bias_initializer=self.bias_initializer,
                                   name=f"1D_{name}")(net)

    def _get_max_pooling_layer(self, input_):
        return keras.layers.MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDES, padding="same",
                                         name="1D_max_pooling_layer")(input_)


if __name__ == "__main__":
    import os

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    shape_ = (92,)
    conf_ = {
        "WITH_BATCH_NORM": True,
        "INCEPTION_FACTOR": 8,
        "DROPOUT": 0.1
    }
    labels = 3
    model1 = InceptionModel1D(input_shape=shape_, model_config=conf_, num_of_output=labels)
    model_ = model1.get_model()
    model_.summary()
