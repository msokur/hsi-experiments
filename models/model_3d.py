import tensorflow as tf 
from tensorflow import keras
import config
from tensorflow.keras import activations

def get_dropout(net, dropout_value=config.DROPOUT):
    if config.WITHOUT_RANDOMNESS:
        net = tf.keras.layers.Dropout(dropout_value, seed=3)(net)
    else:
        net = tf.keras.layers.Dropout(dropout_value)(net)
    return net
    

def get_inizializers():
    kernel_initializer ='glorot_uniform'
    bias_initializer = 'zeros'
    if config.WITHOUT_RANDOMNESS:
        kernel_initializer = keras.initializers.RandomNormal(seed=1337)
        bias_initializer = keras.initializers.Constant(value=0.1)
        
    return kernel_initializer, bias_initializer

def paper_model():
    input_ = tf.keras.layers.Input(
        shape=(config._3D_SIZE[0], config._3D_SIZE[1], config.OUTPUT_SIGNATURE_X_FEATURES), name="title"
    )
    
    kernel_initializer, bias_initializer = get_inizializers()
    
    net = tf.expand_dims(input_, axis=-1)

    net = tf.keras.layers.Conv3D(filters=20, kernel_size=3, padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)
    net = tf.keras.layers.Conv3D(filters=20, kernel_size=(1, 1, 3), strides=(1, 1, 2), padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)
    
    net = tf.keras.layers.Conv3D(filters=20, kernel_size=3, padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)
    net = tf.keras.layers.Conv3D(filters=20, kernel_size=(1, 1, 3), strides=(1, 1, 2), padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)
    
    net = tf.keras.layers.Reshape((net.shape[-2], net.shape[-1]))(net)
    
    net = tf.keras.layers.Conv1D(filters=35, kernel_size=3, padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)
    net = tf.keras.layers.Conv1D(filters=35, kernel_size=3, strides=2, padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)
    
    net = tf.keras.layers.Conv1D(filters=35, kernel_size=3, padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)
    net = tf.keras.layers.Conv1D(filters=35, kernel_size=3, strides=2, padding='valid', activation='relu', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)(net)
    
    net = tf.keras.layers.Flatten()(net)

    #net = tf.keras.layers.Dense(50, activation='relu')(net)
    net = get_dropout(net)
    
    activation = 'sigmoid'
    number = 1
    if config.NUMBER_OF_CLASSES_TO_TRAIN > 2:
        activation = None
        number = config.NUMBER_OF_CLASSES_TO_TRAIN
        
    result = tf.keras.layers.Dense(number, activation=activation)(net)

    model = tf.keras.Model(
        inputs=[input_],
        outputs=[result]
    )

    return model


def inception3d_block(input_, factor=16):
    input_ = tf.expand_dims(input_, axis=-1)
    kernel_initializer, bias_initializer = get_inizializers()


    branch1 = tf.keras.layers.Conv3D(filters=factor*4, kernel_size=1, padding="same", activation='relu', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(input_)

    branch2 = tf.keras.layers.Conv3D(filters=factor*6, kernel_size=1, padding="same", activation='relu', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(input_)
    branch2 = tf.keras.layers.Conv3D(filters=factor*8, kernel_size=3, padding="same", activation='relu', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(branch2)

    branch3 = tf.keras.layers.Conv3D(filters=factor, kernel_size=1, padding="same", activation='relu', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(input_)
    branch3 = tf.keras.layers.Conv3D(filters=factor*2, kernel_size=5, padding="same", activation='relu', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(branch3)

    branch4 = tf.keras.layers.MaxPooling3D(pool_size=3, strides=1, padding="same")(input_)
    branch4 = tf.keras.layers.Conv3D(filters=factor*2, kernel_size=1, activation='relu', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(branch4)

    net = tf.keras.layers.concatenate([branch1, branch2, branch3, branch4])

    return net

def inception3d_model():
    if config.WITHOUT_RANDOMNESS:
        tf.random.set_seed(3)
    
    input_ = tf.keras.layers.Input(
        shape=(config._3D_SIZE[0], config._3D_SIZE[1], config.OUTPUT_SIGNATURE_X_FEATURES), name="title"
    )
    
    kernel_initializer, bias_initializer = get_inizializers()

    
    if config.WITH_BATCH_NORM:
        net = inception3d_block_with_batchnorm(input_, factor=config.INCEPTION_FACTOR)
    else:
        net = inception3d_block(input_, factor=config.INCEPTION_FACTOR)

    net = tf.keras.layers.Flatten()(net)
    #net = tf.keras.layers.Dense(500, activation='relu')(net)
    #net = tf.keras.layers.Dense(250, activation='relu')(net)
    #net = tf.keras.layers.Dropout(config.DROPOUT_VALUE)(net)
    #net = tf.keras.layers.Dense(100, activation='relu')(net)
    #net = tf.keras.layers.Dropout(config.DROPOUT_VALUE)(net)
    #net = tf.keras.layers.Dense(50, activation='relu')(net)
    
    net = get_dropout(net)
    
    result = tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer,
               bias_initializer=bias_initializer)(net)

    model = tf.keras.Model(
        inputs=[input_],
        outputs=[result]
    )

    return model