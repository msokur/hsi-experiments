import tensorflow as tf
import tensorflow.keras as keras

from models.model_randomness import get_dropout, get_initializers


def paper1d_model(shape: tuple, conf: dict, num_of_labels: int):
    input_ = tf.keras.layers.Input(shape=shape, name="title")

    kernel_initializer, bias_initializer = get_initializers()

    net = tf.expand_dims(input_, axis=-1)
    conv_round = range(4)

    for r in conv_round:
        if net.shape[-2] == 1:
            break
        if net.shape[-2] == 2:
            net = get_conv1d(net=net, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                             name=f"1D_{r}", kernel_size=2, strides=1)
            break
        else:
            net = get_conv1d(net=net, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                             name=f"1D_{r}")

    return paper_model_base(input_=input_, net=net, dropout=conf["DROPOUT"], num_of_labels=num_of_labels,
                            kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)


def paper3d_model(shape: tuple, conf: dict, num_of_labels: int):
    input_ = tf.keras.layers.Input(shape=shape, name="title")

    conv_round = range(int(shape[0] / 2))

    kernel_initializer, bias_initializer = get_initializers()

    net = tf.expand_dims(input_, axis=-1)

    for r in conv_round:
        if net.shape[-3] == 2:
            net = get_conv3d(net=net, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                             name=f"3D_{r}", kernel_size=2, strides=1)
            break
        else:
            net = get_conv3d(net=net, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                             name=f"3D_{r}")

    net = tf.keras.layers.Reshape((net.shape[-2], net.shape[-1]))(net)
    print(f"{net.shape[-2]}     {net.shape[-1]}")

    for r in conv_round:
        if net.shape[-2] == 1:
            break
        if net.shape[-2] == 2:
            net = get_conv1d(net=net, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                             name=f"1D_{r}", kernel_size=2, strides=1)
            break
        else:
            net = get_conv1d(net=net, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                             name=f"1D_{r}")

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


def get_conv1d(net, kernel_initializer, bias_initializer, name, kernel_size=3, strides=2):
    net = keras.layers.Conv1D(filters=35, kernel_size=kernel_size, padding='valid', activation='relu',
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                              name=f"{name}_1")(net)
    net = keras.layers.Conv1D(filters=35, kernel_size=kernel_size if net.shape[-2] > 1 else 1, strides=strides,
                              padding='valid', activation='relu',
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                              name=f"{name}_2")(net)

    return net


def get_conv3d(net, kernel_initializer, bias_initializer, name, kernel_size=3, strides=2):
    net = keras.layers.Conv3D(filters=20, kernel_size=kernel_size, padding='valid', activation='relu',
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                              name=f"{name}_1")(net)
    net = keras.layers.Conv3D(filters=20, kernel_size=(1, 1, kernel_size), strides=(1, 1, strides), padding='valid',
                              activation='relu', kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              name=f"{name}_2")(net)

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
