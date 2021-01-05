import tensorflow as tf 
import config

def inception1d_block():
    input_ = tf.keras.layers.Input(
        shape=(config.LAST_NM - config.FIRST_NM, 1), name="title"
    )

    branch1 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding="same", activation='relu')(input_)

    branch2 = tf.keras.layers.Conv1D(filters=96, kernel_size=1, padding="same", activation='relu')(input_)
    branch2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, padding="same", activation='relu')(branch2)

    branch3 = tf.keras.layers.Conv1D(filters=16, kernel_size=1, padding="same", activation='relu')(input_)
    branch3 = tf.keras.layers.Conv1D(filters=32, kernel_size=5, padding="same", activation='relu')(branch3)

    branch4 = tf.keras.layers.MaxPooling1D(pool_size=3, strides=1, padding="same")(input_)
    branch4 = tf.keras.layers.Conv1D(filters=32, kernel_size=1, activation='relu')(branch4)

    net = tf.keras.layers.concatenate([branch1, branch2, branch3, branch4])

    net = tf.keras.layers.Flatten()(net)
    #net = tf.keras.layers.Dense(500, activation='relu')(net)
    #net = tf.keras.layers.Dense(250, activation='relu')(net)
    net = tf.keras.layers.Dense(100, activation='relu')(net)
    net = tf.keras.layers.Dropout(config.DROPOUT_VALUE)(net)
    net = tf.keras.layers.Dense(50, activation='relu')(net)
    net = tf.keras.layers.Dropout(config.DROPOUT_VALUE)(net)
    result = tf.keras.layers.Dense(1, activation='sigmoid')(net)

    model = tf.keras.Model(
        inputs=[input_],
        outputs=[result]
    )

    return model




def get_model():
    model = tf.keras.Sequential()


    model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=3, activation='relu', input_shape=(config.LAST_NM - config.FIRST_NM, 1)))
    #model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=5, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.Conv1D(filters=1, kernel_size=7, activation='relu'))
    #model.add(tf.keras.layers.Dropout(0.5))
    #model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(25, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

if __name__ == '__main__':
    model = inception1d_block()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy())

    model.summary()

