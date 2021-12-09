import tensorflow as tf 
import config
from tensorflow.keras import activations

def paper_model():
    input_ = tf.keras.layers.Input(
        shape=(config._3D_SIZE[0], config._3D_SIZE[1], config.LAST_NM - config.FIRST_NM), name="title"
    )
    
    net = tf.expand_dims(input_, axis=-1)

    net = tf.keras.layers.Conv3D(filters=20, kernel_size=3, padding='valid', activation='relu')(net)
    net = tf.keras.layers.Conv3D(filters=20, kernel_size=(1, 1, 3), strides=(1, 1, 2), padding='valid', activation='relu')(net)
    
    net = tf.keras.layers.Conv3D(filters=20, kernel_size=3, padding='valid', activation='relu')(net)
    net = tf.keras.layers.Conv3D(filters=20, kernel_size=(1, 1, 3), strides=(1, 1, 2), padding='valid', activation='relu')(net)
    
    net = tf.keras.layers.Reshape((net.shape[-2], net.shape[-1]))(net)
    
    net = tf.keras.layers.Conv1D(filters=35, kernel_size=3, padding='valid', activation='relu')(net)
    net = tf.keras.layers.Conv1D(filters=35, kernel_size=3, strides=2, padding='valid', activation='relu')(net)
    
    net = tf.keras.layers.Conv1D(filters=35, kernel_size=3, padding='valid', activation='relu')(net)
    net = tf.keras.layers.Conv1D(filters=35, kernel_size=3, strides=2, padding='valid', activation='relu')(net)
    
    net = tf.keras.layers.Flatten()(net)

    #net = tf.keras.layers.Dense(50, activation='relu')(net)
    net = tf.keras.layers.Dropout(config.DROPOUT_VALUE)(net)
    result = tf.keras.layers.Dense(1, activation='sigmoid')(net)

    model = tf.keras.Model(
        inputs=[input_],
        outputs=[result]
    )

    return model