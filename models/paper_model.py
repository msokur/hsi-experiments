import tensorflow as tf
from models.model_randomness import get_dropout, get_inizializers


def paper1d_model(shape: tuple, conf: dict, num_of_labels: int):
    input_ = tf.keras.layers.Input(shape=shape, name="title")

    kernel_initializer, bias_initializer = get_inizializers()

    net = tf.expand_dims(input_, axis=-1)
    conv_round = range(2)

    for r in conv_round:
        net = get_conv1d(net=net, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

    return paper_model_base(input_=input_, net=net, dropout=conf["DROPOUT"], num_of_labels=num_of_labels,
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)


def paper3d_model(shape: tuple, conf: dict, num_of_labels: int):
    input_ = tf.keras.layers.Input(shape=shape, name="title")

    conv_round = range(int(shape[0] / 2))

    kernel_initializer, bias_initializer = get_inizializers()

    net = tf.expand_dims(input_, axis=-1)

    for r in conv_round:
        net = get_conv3d(net=net, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

    net = tf.keras.layers.Reshape((net.shape[-2], net.shape[-1]))(net)

    for r in conv_round:
        net = get_conv1d(net=net, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)

    return paper_model_base(input_=input_, net=net, dropout=conf["DROPOUT"], num_of_labels=num_of_labels,
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)


def paper_model_base(input_, net, dropout, num_of_labels, kernel_initializer, bias_initializer):
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


def get_conv1d(net, kernel_initializer, bias_initializer):
    net = tf.keras.layers.Conv1D(filters=35, kernel_size=3, padding='valid', activation='relu',
                                 kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)
    net = tf.keras.layers.Conv1D(filters=35, kernel_size=3, strides=2, padding='valid', activation='relu',
                                 kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)

    return net


def get_conv3d(net, kernel_initializer, bias_initializer):
    net = tf.keras.layers.Conv3D(filters=20, kernel_size=3, padding='valid', activation='relu',
                                 kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)
    net = tf.keras.layers.Conv3D(filters=20, kernel_size=(1, 1, 3), strides=(1, 1, 2), padding='valid',
                                 activation='relu', kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer)(net)

    return net


if __name__ == "__main__":
    import os

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    shape_ = (3, 3, 92)
    conf_ = {
        "DROPOUT": 0.1
    }
    labels = 3
    model_ = paper3d_model(shape=shape_, conf=conf_, num_of_labels=labels)
    model_.summary()
