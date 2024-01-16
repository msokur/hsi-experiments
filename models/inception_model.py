import abc

import tensorflow as tf
from tensorflow.keras import activations

from models.model_randomness import get_dropout, get_initializers, set_tf_seed

FILTERS = [4, [6, 8], [1, 2]]
FILTERS_LAST = 2
KERNEL_SIZE = [1, [1, 3], [1, 5]]
KERNEL_SIZE_LAST = 1
POOL_SIZE = 3
POOL_STRIDES = 1


class InceptionModelBase:
    def __init__(self):
        self.num_of_labels = None
        self.kernel_initializer, self.bias_initializer = get_initializers()
        set_tf_seed()

    def get_model(self, shape: tuple, local_config: dict, num_of_labels: int):
        self.local_config = local_config
        self.num_of_labels = num_of_labels

        input_ = tf.keras.layers.Input(
            shape=shape, name="title"
        )

        net = self.inception_block(input_=input_, factor=local_config["INCEPTION_FACTOR"],
                                   with_batch_norm=local_config["WITH_BATCH_NORM"])

        return self.inception_base(input_=input_, net=net)

    def inception_block(self, input_, factor=16, with_batch_norm=False):
        input_ = tf.expand_dims(input_, axis=-1)

        branches = []
        for filter_, kernel_size in zip(FILTERS, KERNEL_SIZE):
            if isinstance(filter_, int):
                branch = self.get_conv_layer(filters=factor * filter_, kernel_size=kernel_size, net=input_)

                if with_batch_norm:
                    branch = self.get_batch_norm(branch)
            else:
                branch = input_
                for filter_sub, kernel_size_sub in zip(filter_, kernel_size):
                    branch = self.get_conv_layer(filters=factor * filter_sub, kernel_size=kernel_size_sub, net=branch)

                    if with_batch_norm:
                        branch = self.get_batch_norm(branch)
            branches.append(branch)

        branch = self.get_max_pooling_layer(input_=input_)
        branch = self.get_conv_layer(filters=factor * FILTERS_LAST, kernel_size=KERNEL_SIZE_LAST, net=branch)

        if with_batch_norm:
            branch = self.get_batch_norm(branch)
        branches.append(branch)

        net = tf.keras.layers.concatenate(branches)

        return net

    def inception_base(self, input_, net):
        net = tf.keras.layers.Flatten()(net)
        net = get_dropout(net=net, dropout_value=self.local_config["DROPOUT"])

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

    @abc.abstractmethod
    def get_conv_layer(self, filters, kernel_size, net):
        pass

    @abc.abstractmethod
    def get_max_pooling_layer(self, input_):
        pass

    @staticmethod
    def get_batch_norm(branch):
        branch = tf.keras.layers.BatchNormalization()(branch)
        branch = tf.keras.layers.Activation(activations.relu)(branch)
        return branch


class InceptionModel3D(InceptionModelBase):
    def get_conv_layer(self, filters, kernel_size, net):
        return tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, padding="same", activation='relu',
                                      kernel_initializer=self.kernel_initializer,
                                      bias_initializer=self.bias_initializer)(net)

    def get_max_pooling_layer(self, input_):
        return tf.keras.layers.MaxPooling3D(pool_size=POOL_SIZE, strides=POOL_STRIDES, padding="same")(input_)


class InceptionModel1D(InceptionModelBase):
    def get_conv_layer(self, filters, kernel_size, net):
        return tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same", activation='relu',
                                      kernel_initializer=self.kernel_initializer,
                                      bias_initializer=self.bias_initializer)(net)

    def get_max_pooling_layer(self, input_):
        return tf.keras.layers.MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDES, padding="same")(input_)


if __name__ == "__main__":
    import os

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    shape_ = (3, 3, 92)
    conf_ = {
        "WITH_BATCH_NORM": False,
        "INCEPTION_FACTOR": 8,
        "DROPOUT": 0.1
    }
    labels = 3
    model1 = InceptionModel3D()
    model_ = model1.get_model(shape=shape_, local_config=conf_, num_of_labels=labels)
    model_.summary()
