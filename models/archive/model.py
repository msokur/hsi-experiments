import tensorflow as tf
import config
from tensorflow.keras import activations
from models.archive.model_3d import get_dropout, get_inizializers


def inception1d_block(input_, factor=16):
    input_ = tf.expand_dims(input_, axis=-1)

    branch1 = tf.keras.layers.Conv1D(filters=factor * 4, kernel_size=1, padding="same", activation='relu')(input_)

    branch2 = tf.keras.layers.Conv1D(filters=factor * 6, kernel_size=1, padding="same", activation='relu')(input_)
    branch2 = tf.keras.layers.Conv1D(filters=factor * 8, kernel_size=3, padding="same", activation='relu')(branch2)

    branch3 = tf.keras.layers.Conv1D(filters=factor, kernel_size=1, padding="same", activation='relu')(input_)
    branch3 = tf.keras.layers.Conv1D(filters=factor * 2, kernel_size=5, padding="same", activation='relu')(branch3)

    branch4 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(input_)
    branch4 = tf.keras.layers.Conv1D(filters=factor * 2, kernel_size=1, activation='relu')(branch4)

    net = tf.keras.layers.concatenate([branch1, branch2, branch3, branch4])

    return net


def inception1d_block_with_batchnorm(input_, factor=16):
    input_ = tf.expand_dims(input_, axis=-1)

    # branch1 = tf.keras.layers.Conv1D(filters=factor*4, kernel_size=1, padding="same", kernel_initializer='he_uniform')(input_)
    branch1 = tf.keras.layers.Conv1D(filters=factor * 4, kernel_size=1, padding="same")(input_)
    branch1 = tf.keras.layers.BatchNormalization()(branch1)
    branch1 = tf.keras.layers.Activation(activations.relu)(branch1)

    branch2 = tf.keras.layers.Conv1D(filters=factor * 6, kernel_size=1, padding="same")(input_)
    branch2 = tf.keras.layers.BatchNormalization()(branch2)
    branch2 = tf.keras.layers.Activation(activations.relu)(branch2)
    branch2 = tf.keras.layers.Conv1D(filters=factor * 8, kernel_size=3, padding="same")(branch2)
    branch2 = tf.keras.layers.BatchNormalization()(branch2)
    branch2 = tf.keras.layers.Activation(activations.relu)(branch2)

    branch3 = tf.keras.layers.Conv1D(filters=factor, kernel_size=1, padding="same")(input_)
    branch3 = tf.keras.layers.BatchNormalization()(branch3)
    branch3 = tf.keras.layers.Activation(activations.relu)(branch3)
    branch3 = tf.keras.layers.Conv1D(filters=factor * 2, kernel_size=5, padding="same")(branch3)
    branch3 = tf.keras.layers.BatchNormalization()(branch3)
    branch3 = tf.keras.layers.Activation(activations.relu)(branch3)

    branch4 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(input_)
    branch4 = tf.keras.layers.Conv1D(filters=factor * 2, kernel_size=1)(branch4)
    branch4 = tf.keras.layers.BatchNormalization()(branch4)
    branch4 = tf.keras.layers.Activation(activations.relu)(branch4)

    net = tf.keras.layers.concatenate([branch1, branch2, branch3, branch4])

    return net


def lstm_block():
    input_ = tf.keras.layers.Input(
        shape=(config.LAST_NM - config.FIRST_NM), name="title"
    )

    net = tf.expand_dims(input_, axis=-1)

    net = inception1d_block(net, factor=config.INCEPTION_FACTOR)

    net = tf.transpose(net, perm=[0, 2, 1])

    net = tf.keras.layers.LSTM(1)(net)

    # net = tf.keras.layers.Dropout(config.DROPOUT_VALUE)(net)
    # result = tf.keras.layers.Dense(1, activation='sigmoid')(net)
    result = tf.keras.layers.Activation('sigmoid')(net)

    model = tf.keras.Model(
        inputs=[input_],
        outputs=[result]
    )

    return model


def inception_model():
    input_ = tf.keras.layers.Input(
        shape=(config.LAST_NM - config.FIRST_NM - 1), name="title"
    )

    if config.WITH_BATCH_NORM:
        net = inception1d_block_with_batchnorm(input_, factor=config.INCEPTION_FACTOR)
    else:
        net = inception1d_block(input_, factor=config.INCEPTION_FACTOR)

    net = tf.keras.layers.Flatten()(net)
    # net = tf.keras.layers.Dense(500, activation='relu')(net)
    # net = tf.keras.layers.Dense(250, activation='relu')(net)
    # net = tf.keras.layers.Dropout(config.DROPOUT_VALUE)(net)
    # net = tf.keras.layers.Dense(100, activation='relu')(net)
    # net = tf.keras.layers.Dropout(config.DROPOUT_VALUE)(net)
    net = tf.keras.layers.Dense(50, activation='relu')(net)
    net = tf.keras.layers.Dropout(config.DROPOUT)(net)
    result = tf.keras.layers.Dense(1, activation='sigmoid')(net)

    model = tf.keras.Model(
        inputs=[input_],
        outputs=[result]
    )

    return model


def paper1d_model():
    input_ = tf.keras.layers.Input(
        shape=(config.LAST_NM - config.FIRST_NM), name="title")

    kernel_initializer, bias_initializer = get_inizializers()

    net = tf.expand_dims(input_, axis=-1)
    conv_round = range(2)

    for r in conv_round:
        net = tf.keras.layers.Conv1D(filters=35, kernel_size=3, padding='valid', activation='relu',
                                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)
        net = tf.keras.layers.Conv1D(filters=35, kernel_size=3, strides=2, padding='valid', activation='relu',
                                     kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)

    net = tf.keras.layers.Flatten()(net)

    # net = tf.keras.layers.Dense(50, activation='relu')(net)
    net = get_dropout(net)

    activation = 'sigmoid'
    number = 1
    if config.NUMBER_OF_CLASSES_TO_TRAIN > 2:
        activation = None
        number = config.NUMBER_OF_CLASSES_TO_TRAIN

    result = tf.keras.layers.Dense(number, activation=activation, kernel_initializer=kernel_initializer,
                                   bias_initializer=bias_initializer)(net)

    model = tf.keras.Model(
        inputs=[input_],
        outputs=[result]
    )

    return model


def get_model():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=3, activation='relu',
                                     input_shape=(config.LAST_NM - config.FIRST_NM, 1)))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=5, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=7, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.5))
    # model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(25, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model


if __name__ == '__main__':
    model_ = inception1d_block()

    model_.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy())

    model_.summary()
