import tensorflow as tf
from tensorflow.keras import activations

from models.model_randomness import get_dropout, get_initializers

FILTERS = [4, [6, 8], [1, 2]]
FILTERS_LAST = 2
KERNEL_SIZE = [1, [1, 3], [1, 5]]
KERNEL_SIZE_LAST = 1
POOL_SIZE = 3
POOL_STRIDES = 1


def inception1d_model(shape: tuple, conf: dict, num_of_labels: int):
    input_ = tf.keras.layers.Input(
        shape=shape, name="title"
    )

    kernel_initializer, bias_initializer = get_initializers()

    net = inception1d_block(input_, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                            factor=conf["INCEPTION_FACTOR"], with_batch_norm=conf["WITH_BATCH_NORM"])

    return inception_base(input_=input_, net=net, dropout=conf["DROPOUT"], num_of_labels=num_of_labels,
                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)


def inception3d_model(shape: tuple, conf: dict, num_of_labels: int):
    input_ = tf.keras.layers.Input(
        shape=shape, name="title"
    )

    kernel_initializer, bias_initializer = get_initializers()

    net = inception3d_block(input_, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                            factor=conf["INCEPTION_FACTOR"], with_batch_norm=conf["WITH_BATCH_NORM"])

    return inception_base(input_=input_, net=net, dropout=conf["DROPOUT"], num_of_labels=num_of_labels,
                          kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)


def inception_base(input_, net, dropout, num_of_labels, kernel_initializer, bias_initializer):
    net = tf.keras.layers.Flatten()(net)
    net = get_dropout(net=net, dropout_value=dropout)

    activation = 'sigmoid'
    number = 1
    if num_of_labels > 2:
        activation = None
        number = num_of_labels
    result = tf.keras.layers.Dense(number, activation=activation, kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer)(net)

    model = tf.keras.Model(
        inputs=[input_],
        outputs=[result]
    )

    return model


def inception1d_block(input_, kernel_initializer, bias_initializer, factor=16, with_batch_norm=False):
    input_ = tf.expand_dims(input_, axis=-1)

    branches = []
    for filter_, kernel_size in zip(FILTERS, KERNEL_SIZE):
        if isinstance(filter_, int):
            branch = get_conv1d(filters=factor * filter_, kernel_size=kernel_size,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer, net=input_)
            if with_batch_norm:
                branch = get_batch_norm(branch)
        else:
            branch = input_
            for filter_sub, kernel_size_sub in zip(filter_, kernel_size):
                branch = get_conv1d(filters=factor * filter_sub, kernel_size=kernel_size_sub,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer, net=branch)
                if with_batch_norm:
                    branch = get_batch_norm(branch)
        branches.append(branch)

    branch = tf.keras.layers.MaxPooling1D(pool_size=POOL_SIZE, strides=POOL_STRIDES, padding="same")(input_)
    branch = get_conv1d(filters=factor * FILTERS_LAST, kernel_size=KERNEL_SIZE_LAST,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer, net=branch)
    if with_batch_norm:
        branch = get_batch_norm(branch)
    branches.append(branch)

    net = tf.keras.layers.concatenate(branches)

    return net


def inception3d_block(input_, kernel_initializer, bias_initializer, factor=16, with_batch_norm=False):
    input_ = tf.expand_dims(input_, axis=-1)

    branches = []
    for filter_, kernel_size in zip(FILTERS, KERNEL_SIZE):
        if isinstance(filter_, int):
            branch = get_conv3d(filters=factor * filter_, kernel_size=kernel_size,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer, net=input_)
            if with_batch_norm:
                branch = get_batch_norm(branch)
        else:
            branch = input_
            for filter_sub, kernel_size_sub in zip(filter_, kernel_size):
                branch = get_conv3d(filters=factor * filter_sub, kernel_size=kernel_size_sub,
                                    kernel_initializer=kernel_initializer,
                                    bias_initializer=bias_initializer, net=branch)
                if with_batch_norm:
                    branch = get_batch_norm(branch)
        branches.append(branch)

    branch = tf.keras.layers.MaxPooling3D(pool_size=POOL_SIZE, strides=POOL_STRIDES, padding="same")(input_)
    branch = get_conv3d(filters=factor * FILTERS_LAST, kernel_size=KERNEL_SIZE_LAST,
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer, net=branch)
    if with_batch_norm:
        branch = get_batch_norm(branch)
    branches.append(branch)

    net = tf.keras.layers.concatenate(branches)

    return net


def get_conv1d(filters, kernel_size, kernel_initializer, bias_initializer, net):
    return tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same", activation='relu',
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=bias_initializer)(net)


def get_conv3d(filters, kernel_size, kernel_initializer, bias_initializer, net):
    return tf.keras.layers.Conv3D(filters=filters, kernel_size=kernel_size, padding="same", activation='relu',
                                  kernel_initializer=kernel_initializer,
                                  bias_initializer=bias_initializer)(net)


def get_batch_norm(branch):
    branch = tf.keras.layers.BatchNormalization()(branch)
    branch = tf.keras.layers.Activation(activations.relu)(branch)
    return branch


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
    model_ = inception3d_model(shape=shape_, conf=conf_, num_of_labels=labels)
    model_.summary()
